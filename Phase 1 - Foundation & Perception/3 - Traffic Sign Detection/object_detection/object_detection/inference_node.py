#!/usr/bin/env python3

"""
inference_node.py.

ROS 2 node - node name: inference_node

Subscribes  (topic names are parameters, see config/object_detection_params.yaml)
----------
/oak/rgb/image_raw        sensor_msgs/Image   raw OAK-D colour image
/oak/stereo/image_raw     sensor_msgs/Image   OAK-D depth image (millimetres)

Publishes
---------
/detections               object_detection_msgs/DetectionArray
                          one entry per box, distance in metres (-1 = unknown)
/detections/image         sensor_msgs/Image   bgr8 image with the boxes drawn;
                          only computed while something subscribes

Detection runs with or without depth: without a depth image yet, distances
are reported as -1. Frame rate and inference time are logged periodically
instead of once per frame.
"""

import os
import time

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
import numpy as np
from object_detection_msgs.msg import Detection
from object_detection_msgs.msg import DetectionArray
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from .depth_estimator import bbox_distance
from .detector import Detector
from .visualization import Visualizer


class InferenceNode(Node):
    """Detect traffic signs in every camera frame and publish them."""

    def __init__(self):
        """Declare parameters, load the model, create publishers/subscribers."""
        super().__init__("inference_node")

        self.bridge = CvBridge()

        self.latest_depth = None

        # --------------------------------------------------------
        # Parameters (see config/object_detection_params.yaml)
        # --------------------------------------------------------

        self.declare_parameter("model_path", "")
        self.declare_parameter("confidence", 0.50)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("image_size", 640)
        self.declare_parameter("device", "")
        self.declare_parameter("half", False)
        self.declare_parameter("depth_roi_half_size", 3)
        self.declare_parameter("show_window", False)
        self.declare_parameter("log_detections", False)
        self.declare_parameter("stats_period_sec", 5.0)
        self.declare_parameter("rgb_topic", "/oak/rgb/image_raw")
        self.declare_parameter("depth_topic", "/oak/stereo/image_raw")
        self.declare_parameter("detections_topic", "/detections")
        self.declare_parameter("annotated_topic", "/detections/image")

        def param(name):
            return self.get_parameter(name).value

        self.depth_roi_half_size = param("depth_roi_half_size")
        self.show_window = param("show_window")
        self.log_detections = param("log_detections")

        # --------------------------------------------------------
        # Load YOLO Model
        # --------------------------------------------------------

        # Empty model_path -> bundled models/last.pt
        model_path = param("model_path")
        if not model_path:
            model_path = os.path.join(
                get_package_share_directory("object_detection"),
                "models",
                "last.pt"
            )

        self.detector = Detector(
            model_path=model_path,
            confidence=param("confidence"),
            iou=param("iou"),
            image_size=param("image_size"),
            device=param("device") or None,
            half=param("half"),
        )

        self.visualizer = Visualizer()

        self.detection_pub = self.create_publisher(
            DetectionArray,
            param("detections_topic"),
            10,
        )

        self.annotated_pub = self.create_publisher(
            Image,
            param("annotated_topic"),
            1,
        )

        # best effort, keep only the newest frame: when inference is slower
        # than the camera, older frames are dropped instead of waiting in a
        # queue (a queue of 5 added ~200 ms of latency on the Jetson)
        latest_only = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            Image,
            param("rgb_topic"),
            self.image_callback,
            latest_only,
        )

        self.create_subscription(
            Image,
            param("depth_topic"),
            self.depth_callback,
            latest_only,
        )

        self.frames = 0
        self.detected = 0
        self.infer_time = 0.0
        self.warned_depth_size = False
        self.create_timer(float(param("stats_period_sec")), self.log_stats)

        self.get_logger().info(f"Subscribed to {param('rgb_topic')}")

    def image_callback(self, msg):
        """Detect objects in one frame and publish them."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001 - bad frames must not kill the node
            self.get_logger().error(f"cv_bridge failed: {exc}")
            return

        # Camera mounted upside down
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # ---------------------------------------------
        # Run YOLO
        # ---------------------------------------------

        t0 = time.perf_counter()
        detections = self.detector.detect(frame)
        self.infer_time += time.perf_counter() - t0
        self.frames += 1
        self.detected += len(detections)

        # Estimate distance for every detection
        depth = self.latest_depth
        if depth is not None and depth.shape[:2] != frame.shape[:2] \
                and not self.warned_depth_size:
            self.get_logger().warn(
                f"Depth image {depth.shape[1]}x{depth.shape[0]} differs from "
                f"colour image {frame.shape[1]}x{frame.shape[0]}: box centres "
                "are scaled to the depth image, which is only correct if the "
                "driver aligns depth to the colour camera.")
            self.warned_depth_size = True

        for det in detections:
            det["distance"] = (
                bbox_distance(depth, det["bbox"], frame.shape,
                              self.depth_roi_half_size)
                if depth is not None else None
            )

        self.publish_detections(detections, msg.header)

        if self.log_detections:
            for det in detections:
                dist = (
                    f"{det['distance']:.2f} m"
                    if det["distance"] is not None
                    else "N/A"
                )
                self.get_logger().info(
                    f"{det['class_name']:20s}"
                    f" Conf={det['confidence']:.2f}"
                    f" Dist={dist}"
                )

        # ---------------------------------------------
        # Draw detections - only when someone looks
        # ---------------------------------------------

        watched = self.annotated_pub.get_subscription_count() > 0
        if not (watched or self.show_window):
            return

        annotated = self.visualizer.draw(frame, detections)

        if watched:
            out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out.header = msg.header
            self.annotated_pub.publish(out)

        if self.show_window:
            cv2.imshow("Detections", annotated)
            cv2.waitKey(1)

    def depth_callback(self, msg):
        """Keep the newest depth image, rotated like the colour image."""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"cv_bridge failed on depth: {exc}")
            return

        # Rotate depth exactly like RGB
        self.latest_depth = cv2.rotate(depth, cv2.ROTATE_180)

        if self.show_window:
            depth_vis = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow("Depth", depth_vis)
            cv2.waitKey(1)

    def publish_detections(self, detections, header):
        """Publish one DetectionArray for a frame."""
        msg = DetectionArray()
        msg.header = header

        for det in detections:

            detection = Detection()

            detection.header = header

            detection.class_name = det["class_name"]
            detection.class_id = det["class_id"]

            detection.confidence = float(det["confidence"])

            if det["distance"] is not None:
                detection.distance = float(det["distance"])
            else:
                detection.distance = -1.0

            x1, y1, x2, y2 = det["bbox"]

            detection.xmin = x1
            detection.ymin = y1
            detection.xmax = x2
            detection.ymax = y2

            msg.detections.append(detection)

        self.detection_pub.publish(msg)

    def log_stats(self):
        """Log throughput periodically instead of once per frame."""
        if self.frames:
            ms = 1000.0 * self.infer_time / self.frames
            depth = "yes" if self.latest_depth is not None else "NO (distances = -1)"
            self.get_logger().info(
                f"{self.frames} frames, {ms:.1f} ms/frame inference, "
                f"{self.detected} detections, depth: {depth}")
        else:
            self.get_logger().warn("No frames received - is the camera publishing?")
        self.frames = self.detected = 0
        self.infer_time = 0.0


def main(args=None):
    """Entry point for `ros2 run object_detection inference_node`."""
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
