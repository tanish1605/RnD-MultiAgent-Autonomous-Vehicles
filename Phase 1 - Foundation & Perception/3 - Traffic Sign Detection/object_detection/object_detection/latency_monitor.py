#!/usr/bin/env python3

"""
latency_monitor.py.

ROS 2 node - node name: latency_monitor

Measures how late the outputs of the perception nodes are, compared with the
camera frame they were computed from. Every watched message must carry the
camera image's header (inference_node and segmentation_node both copy it), so

    latency = time the message arrives here - header.stamp of the camera frame

covers capture, transfer, inference and publishing. Message types are looked
up automatically, so any stamped topic can be watched.

Every report_period_sec it logs, per topic: rate, mean, p50 / p95 / p99 and
max latency. p95 / p99 show jitter, which matters more to a control loop than
the mean. On Ctrl+C it prints the totals and, with csv_path set, writes every
sample to a CSV file for plotting.

Usage
-----
  ros2 run object_detection latency_monitor
  ros2 run object_detection latency_monitor --ros-args
      -p topics:="['/detections', '/segmentation/mask']" -p csv_path:=/tmp/latency.csv
  (all on one line)

Run tegrastats alongside it to log GPU load:
  tegrastats --interval 1000 --logfile /tmp/tegrastats.log
"""

import csv

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message

from .utils import latency_summary


class LatencyMonitor(Node):
    """Subscribe to stamped topics and report their latency."""

    def __init__(self):
        """Declare parameters and start looking for the topics."""
        super().__init__("latency_monitor")

        self.declare_parameter("topics", ["/detections"])
        self.declare_parameter("camera_topic", "/oak/rgb/image_raw")
        self.declare_parameter("report_period_sec", 5.0)
        self.declare_parameter("csv_path", "")

        self.camera_topic = self.get_parameter("camera_topic").value
        watched = list(self.get_parameter("topics").value)
        self.csv_path = self.get_parameter("csv_path").value

        # camera first: its rate is the reference for the other topics
        self.pending = [self.camera_topic] + [t for t in watched if t != self.camera_topic]
        self.window = {}    # topic -> ([latency_s], [arrival_s]) since last report
        self.total = {}     # topic -> ([latency_s], [arrival_s]) whole run
        self.samples = []   # (topic, arrival_s, stamp_s, latency_s) for the CSV
        self.subs = []

        self.create_timer(1.0, self.subscribe_pending)
        self.create_timer(float(self.get_parameter("report_period_sec").value),
                          self.report)

    def subscribe_pending(self):
        """Subscribe to every topic that has appeared (types are looked up)."""
        if not self.pending:
            return
        types = dict(self.get_topic_names_and_types())
        for topic in list(self.pending):
            if topic not in types:
                continue
            msg_type = get_message(types[topic][0])
            if "header" not in msg_type.get_fields_and_field_types():
                self.get_logger().error(f"{topic} has no header - cannot measure latency")
                self.pending.remove(topic)
                continue
            self.subs.append(self.create_subscription(
                msg_type, topic, lambda msg, t=topic: self.on_message(t, msg),
                qos_profile_sensor_data))
            self.window[topic] = ([], [])
            self.total[topic] = ([], [])
            self.pending.remove(topic)
            self.get_logger().info(f"Watching {topic} ({types[topic][0]})")

    def on_message(self, topic, msg):
        """Record one message's latency."""
        now = self.get_clock().now().nanoseconds * 1e-9
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        latency = now - stamp
        for store in (self.window, self.total):
            store[topic][0].append(latency)
            store[topic][1].append(now)
        if self.csv_path:
            self.samples.append((topic, now, stamp, latency))

    def _lines(self, store):
        cam = latency_summary(*store[self.camera_topic]) if self.camera_topic in store else None
        lines = []
        for topic, (lat, arr) in store.items():
            s = latency_summary(lat, arr)
            if s is None:
                lines.append(f"  {topic:28s} no messages")
                continue
            share = ""
            if topic != self.camera_topic and cam and cam["count"]:
                share = f"  ({100.0 * s['count'] / cam['count']:5.1f} % of camera frames)"
            lines.append(
                f"  {topic:28s} {s['rate_hz']:5.1f} Hz  latency ms: mean {s['mean_ms']:6.1f}"
                f"  p50 {s['p50_ms']:6.1f}  p95 {s['p95_ms']:6.1f}  p99 {s['p99_ms']:6.1f}"
                f"  max {s['max_ms']:6.1f}{share}")
        return lines

    def report(self):
        """Log the window's statistics and start a new window."""
        if self.pending:
            self.get_logger().info(f"Waiting for topics: {', '.join(self.pending)}")
        if self.window:
            self.get_logger().info("\n" + "\n".join(self._lines(self.window)))
        for topic in self.window:
            self.window[topic] = ([], [])

    def finish(self):
        """Print whole-run totals and write the CSV."""
        if self.total:
            print("\nWhole run:\n" + "\n".join(self._lines(self.total)))
        if self.csv_path and self.samples:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["topic", "arrival_s", "stamp_s", "latency_ms"])
                for topic, arrival, stamp, latency in self.samples:
                    writer.writerow([topic, f"{arrival:.6f}", f"{stamp:.6f}",
                                     f"{1000.0 * latency:.3f}"])
            print(f"Wrote {len(self.samples)} samples to {self.csv_path}")


def main(args=None):
    """Entry point for `ros2 run object_detection latency_monitor`."""
    rclpy.init(args=args)
    node = LatencyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
