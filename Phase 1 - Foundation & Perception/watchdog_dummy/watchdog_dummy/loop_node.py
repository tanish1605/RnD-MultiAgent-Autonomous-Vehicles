#!/usr/bin/env python3

import json
import socket
import time

import rclpy
from rclpy.node import Node


def fnv1a32(data: bytes) -> int:
    """FNV-1a 32-bit checksum; must match the watchdog implementation."""
    value = 0x811C9DC5
    for byte in data:
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def heartbeat_checksum(message: dict) -> int:
    healthy = 1 if message["isHealthy"] else 0
    payload = (
        f'{message["id"]}|'
        f'{message["timestamp_ns"]}|'
        f'{healthy}|'
        f'{message.get("msg", "")}'
    )
    return fnv1a32(payload.encode("utf-8"))


class DummyLoopNode(Node):
    def __init__(self):
        super().__init__("watchdog_dummy_loop")

        self.declare_parameter("node_id", 1)
        self.declare_parameter("socket_path", "/tmp/master_unix")
        self.declare_parameter("loop_period_sec", 1.0)
        self.declare_parameter("heartbeat_period_sec", 0.5)
        self.declare_parameter("heartbeat_enabled", True)
        self.declare_parameter("is_healthy", True)
        self.declare_parameter("status_message", "OK")

        self.node_id = int(self.get_parameter("node_id").value)
        self.socket_path = str(self.get_parameter("socket_path").value)
        self.heartbeat_enabled = bool(
            self.get_parameter("heartbeat_enabled").value
        )
        self.is_healthy = bool(self.get_parameter("is_healthy").value)
        self.status_message = str(
            self.get_parameter("status_message").value
        )

        loop_period = float(self.get_parameter("loop_period_sec").value)
        heartbeat_period = float(
            self.get_parameter("heartbeat_period_sec").value
        )

        self.iteration = 0
        self.last_heartbeat = 0.0
        self.loop_timer = self.create_timer(loop_period, self.loop_once)
        self.heartbeat_timer = self.create_timer(
            heartbeat_period, self.send_heartbeat
        )

        self.get_logger().info(
            f"Dummy loop node started (id={self.node_id}, "
            f"socket={self.socket_path})"
        )

    def loop_once(self):
        self.iteration += 1
        self.get_logger().info(f"Loop iteration: {self.iteration}")

    def send_heartbeat(self):
        if not self.heartbeat_enabled:
            return

        message = {
            "id": self.node_id,
            "timestamp_ns": time.time_ns(),
            "isHealthy": self.is_healthy,
            "msg": self.status_message,
        }
        message["checksum"] = heartbeat_checksum(message)

        payload = (
            json.dumps(message, separators=(",", ":")) + "\n"
        ).encode("utf-8")

        try:
            # One connection per heartbeat, matching the watchdog's
            # newline-delimited JSON socket protocol.
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                sock.connect(self.socket_path)
                sock.sendall(payload)

            now = time.monotonic()
            if now - self.last_heartbeat >= 2.0:
                self.get_logger().info(
                    f"Heartbeat sent: healthy={self.is_healthy}, "
                    f"status={self.status_message}"
                )
                self.last_heartbeat = now

        except OSError as exc:
            self.get_logger().warning(
                f"Could not send heartbeat to {self.socket_path}: {exc}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = DummyLoopNode()
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
