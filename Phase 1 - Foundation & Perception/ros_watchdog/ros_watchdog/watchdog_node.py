
#!/usr/bin/env python3

import json
import os
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path

import yaml
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

def fnv1a32(data: bytes) -> int:
    """FNV-1a 32-bit checksum."""
    value = 0x811C9DC5
    for byte in data:
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def heartbeat_checksum(message: dict) -> int:
    """
    Sender and receiver must use exactly this canonical format.
    """
    healthy = 1 if message["isHealthy"] else 0
    payload = (
        f'{message["id"]}|'
        f'{message["timestamp_ns"]}|'
        f'{healthy}|'
        f'{message.get("msg", "")}'
    )
    return fnv1a32(payload.encode("utf-8"))


class ROSWatchdog(Node):

    def __init__(self):
        super().__init__("ros_watchdog")

        config_path = os.environ.get("WATCHDOG_CONFIG")

        if config_path is None:
            config_path = os.path.join(
                get_package_share_directory("ros_watchdog"),
                "config",
                "watchdog.yaml",
            )

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        settings = config["watchdog"]
        self.socket_path = settings["socket_path"]
        self.check_period = float(settings.get("check_period_sec", 0.2))
        self.default_timeout = float(
            settings.get("heartbeat_timeout_sec", 2.0)
        )
        self.restart_delay = float(
            settings.get("restart_delay_sec", 1.0)
        )
        self.max_restarts = int(settings.get("max_restarts", 5))

        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.processes = {}
        self.last_seen = {}
        self.restart_counts = {}
        self.healthy = {}
        self.restart_in_progress = set()

        self.nodes_config = {
            int(item["id"]): item for item in config["nodes"]
        }

        for node_id in self.nodes_config:
            self.last_seen[node_id] = time.monotonic()
            self.restart_counts[node_id] = 0
            self.healthy[node_id] = False

        # Start only processes that this watchdog owns.
        for node_id, item in self.nodes_config.items():
            if item.get("restart", True):
                self._start_process(node_id)

        self.server_thread = threading.Thread(
            target=self._socket_server, daemon=True
        )
        self.server_thread.start()

        self.timer = self.create_timer(
            self.check_period, self._check_health
        )

        self.get_logger().info("ROS Watchdog started")

    def _start_process(self, node_id):
        item = self.nodes_config[node_id]

        with self.lock:
            current = self.processes.get(node_id)
            if current is not None and current.poll() is None:
                return

            if self.restart_counts[node_id] >= self.max_restarts:
                self.get_logger().error(
                    f'{item["name"]}: restart limit reached'
                )
                return

            command = item["command"]
            if isinstance(command, str):
                command = shlex.split(command)

            try:
                process = subprocess.Popen(
                    command,
                    start_new_session=True,
                )
                self.processes[node_id] = process
                self.restart_counts[node_id] += 1
                self.last_seen[node_id] = time.monotonic()
                self.healthy[node_id] = False

                self.get_logger().info(
                    f'{item["name"]}: started PID {process.pid}'
                )
            except (OSError, ValueError) as exc:
                self.get_logger().error(
                    f'{item["name"]}: launch failed: {exc}'
                )

    def _socket_server(self):
        socket_path = self.socket_path
        Path(socket_path).parent.mkdir(
            parents=True, exist_ok=True
        )

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket = server

        try:
            if os.path.exists(socket_path):
                os.unlink(socket_path)

            server.bind(socket_path)
            os.chmod(socket_path, 0o600)
            server.listen(10)
            server.settimeout(0.5)

            while not self.stopping.is_set():
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                threading.Thread(
                    target=self._handle_client,
                    args=(conn,),
                    daemon=True,
                ).start()

        except OSError as exc:
            if not self.stopping.is_set():
                self.get_logger().error(
                    f"Socket server failed: {exc}"
                )
        finally:
            server.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)

    def _handle_client(self, conn):
        # Protocol: one UTF-8 JSON object per newline.
        buffer = b""

        try:
            with conn:
                conn.settimeout(1.0)

                while not self.stopping.is_set():
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        break

                    if not chunk:
                        break

                    buffer += chunk
                    if len(buffer) > 65536:
                        break

                    while b"\n" in buffer:
                        raw, buffer = buffer.split(b"\n", 1)

                        try:
                            message = json.loads(raw.decode("utf-8"))
                            self._process_heartbeat(message)
                        except (
                            ValueError,
                            UnicodeDecodeError,
                            KeyError,
                            TypeError,
                        ) as exc:
                            self.get_logger().warning(
                                f"Invalid heartbeat: {exc}"
                            )
        except OSError:
            pass

    def _process_heartbeat(self, message):
        node_id = int(message["id"])

        if node_id not in self.nodes_config:
            self.get_logger().warning(
                f"Unknown heartbeat ID: {node_id}"
            )
            return

        expected = heartbeat_checksum(message)
        received = int(message["checksum"])

        if received != expected:
            self.get_logger().warning(
                f"Checksum mismatch for ID {node_id}"
            )
            return

        # Use receiver time for timeout checks. This avoids
        # clock-domain and clock-synchronization problems.
        now = time.monotonic()
        is_healthy = bool(message["isHealthy"])

        with self.lock:
            self.last_seen[node_id] = now
            self.healthy[node_id] = is_healthy

        if not is_healthy:
            self.get_logger().error(
                f'ID {node_id} reports unhealthy: '
                f'{message.get("msg", "")}'
            )

    def _check_health(self):
        now = time.monotonic()

        with self.lock:
            snapshot = list(self.nodes_config.items())

        for node_id, item in snapshot:
            timeout = float(
                item.get(
                    "heartbeat_timeout_sec",
                    self.default_timeout,
                )
            )

            with self.lock:
                last_seen = self.last_seen[node_id]
                is_healthy = self.healthy[node_id]
                process = self.processes.get(node_id)
                restarting = node_id in self.restart_in_progress

            process_exited = (
                process is not None and process.poll() is not None
            )
            timed_out = now - last_seen > timeout
            failed = process_exited or timed_out or not is_healthy

            if failed and item.get("restart", True) and not restarting:
                reason = (
                    "process exited" if process_exited
                    else "heartbeat timeout" if timed_out
                    else "unhealthy heartbeat"
                )

                self.get_logger().warning(
                    f'{item["name"]}: {reason}'
                )

                threading.Thread(
                    target=self._recover,
                    args=(node_id,),
                    daemon=True,
                ).start()

    def _recover(self, node_id):
        with self.lock:
            if node_id in self.restart_in_progress:
                return
            self.restart_in_progress.add(node_id)

        try:
            item = self.nodes_config[node_id]

            with self.lock:
                process = self.processes.get(node_id)

            # Terminate only the process launched by this watchdog.
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

            with self.lock:
                self.healthy[node_id] = False
                self.last_seen[node_id] = time.monotonic()

            if self.stopping.wait(self.restart_delay):
                return

            self._start_process(node_id)

        finally:
            with self.lock:
                self.restart_in_progress.discard(node_id)

    def destroy_node(self):
        self.stopping.set()

        if hasattr(self, "server_socket"):
            self.server_socket.close()

        with self.lock:
            processes = list(self.processes.values())

        for process in processes:
            if process.poll() is None:
                process.terminate()

        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ROSWatchdog()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
