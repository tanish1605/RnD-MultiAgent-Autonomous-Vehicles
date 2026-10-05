#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point

import tf2_ros
import tf2_geometry_msgs

import math

from obstacle_msgs.msg import Obstacle
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class ObstacleDetector(Node):

    def __init__(self):
        super().__init__("obstacle_detector")

        # =============================
        # TF SETUP
        # =============================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # =============================
        # SUBSCRIBER
        # =============================
        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            scan_qos
        )

        # =============================
        # PUBLISHERS
        # =============================
        self.obstacle_pub = self.create_publisher(
            Obstacle,
            "/front_obstacle",
            10
        )

        self.marker_pub = self.create_publisher(
            Marker,
            "/obstacle_marker",
            10
        )

        # =============================
        # PARAMETERS (ONLY SOURCE OF TRUTH)
        # =============================

        # TOTAL FOV = 90° (±45°)
        self.front_fov = math.radians(20)

        # OPTIONAL: adjust if LiDAR forward is not 0°
        # set to math.pi if your LiDAR is rotated 180°
        self.fov_offset = 0.0

        self.cluster_threshold = 0.20

        self.get_logger().info("Obstacle detector running")

    # =========================================================
    # MAIN CALLBACK
    # =========================================================
    def scan_callback(self, scan):

        points = []

        # -----------------------------
        # LASER → XY + FOV FILTER
        # -----------------------------
        for i, r in enumerate(scan.ranges):

            angle = scan.angle_min + i * scan.angle_increment

            # APPLY OFFSET (fixes rotated LiDAR cases)
            angle = angle + self.fov_offset

            # -----------------------------
            # SINGLE SOURCE OF TRUTH FOV
            # -----------------------------
            if abs(angle) > self.front_fov:
                continue

            if scan.range_min < r < scan.range_max:

                x = r * math.cos(angle)
                y = r * math.sin(angle)

                points.append((x, y))

        if len(points) < 5:
            return

        # -----------------------------
        # CLUSTERING
        # -----------------------------
        clusters = []
        current = []

        for p in points:

            if not current:
                current.append(p)
                continue

            last = current[-1]
            dist = math.hypot(p[0] - last[0], p[1] - last[1])

            if dist < self.cluster_threshold:
                current.append(p)
            else:
                if len(current) > 5:
                    clusters.append(current)
                current = [p]

        if len(current) > 5:
            clusters.append(current)

        if not clusters:
            return

        # -----------------------------
        # NEAREST OBSTACLE
        # -----------------------------
        obstacle = min(
            clusters,
            key=lambda c: math.hypot(c[len(c)//2][0], c[len(c)//2][1])
        )

        left = obstacle[0]
        right = obstacle[-1]

        cx = (left[0] + right[0]) / 2.0
        cy = (left[1] + right[1]) / 2.0

        width = math.hypot(right[0] - left[0], right[1] - left[1])
        distance = math.hypot(cx, cy)

        # -----------------------------
        # TF TRANSFORM
        # -----------------------------
        center_odom = self.transform_point(cx, cy)
        left_odom = self.transform_point(left[0], left[1])
        right_odom = self.transform_point(right[0], right[1])

        self.publish_marker()

        if center_odom is None:
            return

        # -----------------------------
        # MESSAGE
        # -----------------------------
        msg = Obstacle()

        msg.center_x = center_odom[0]
        msg.center_y = center_odom[1]

        msg.distance = distance
        msg.width = width

        msg.left_x = left_odom[0] if left_odom else 0.0
        msg.left_y = left_odom[1] if left_odom else 0.0

        msg.right_x = right_odom[0] if right_odom else 0.0
        msg.right_y = right_odom[1] if right_odom else 0.0

        msg.angle = math.atan2(cy, cx)

        self.obstacle_pub.publish(msg)

    # =========================================================
    # TF TRANSFORM
    # =========================================================
    def transform_point(self, x, y):

        p = tf2_geometry_msgs.PointStamped()
        p.header.frame_id = "lidar_link"
        p.header.stamp = self.get_clock().now().to_msg()

        p.point.x = x
        p.point.y = y
        p.point.z = 0.0

        try:
            tf = self.tf_buffer.lookup_transform(
                "odom",
                "lidar_link",
                rclpy.time.Time()
            )

            out = tf2_geometry_msgs.do_transform_point(p, tf)

            return (out.point.x, out.point.y)

        except Exception:
            return None

    # =========================================================
    # VISUALIZATION (FULLY CONSISTENT WITH FOV)
    # =========================================================
    def publish_marker(self):

        m2 = Marker()
        m2.header.frame_id = "lidar_link"
        m2.header.stamp = self.get_clock().now().to_msg()

        m2.ns = "fov"
        m2.id = 1
        m2.type = Marker.LINE_LIST
        m2.action = Marker.ADD

        m2.scale.x = 0.02

        m2.color.r = 0.0
        m2.color.g = 1.0
        m2.color.b = 0.0
        m2.color.a = 1.0

        length = 5.0

        center = Point()
        center.x = 0.0
        center.y = 0.0

        # LEFT boundary
        left_pt = Point()
        left_pt.x = length * math.cos(self.front_fov)
        left_pt.y = length * math.sin(self.front_fov)

        # RIGHT boundary
        right_pt = Point()
        right_pt.x = length * math.cos(-self.front_fov)
        right_pt.y = length * math.sin(-self.front_fov)

        m2.points = [center, left_pt, center, right_pt]

        self.marker_pub.publish(m2)


# =========================================================
# MAIN
# =========================================================
def main():
    rclpy.init()
    node = ObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
