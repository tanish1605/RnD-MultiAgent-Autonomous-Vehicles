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

sign_geometry:=true (with a segmentation model, e.g. YOLO26-seg) turns on the
shape-aware localisation: the sign's outline is fitted with its known shape
and size (config/sign_geometry.yaml) and fused with stereo depth; the
published distance is then the fused one. geometry_log:=<file.csv> writes
every method's estimate per detection, for comparing them.
"""

import csv
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
from sensor_msgs.msg import CameraInfo, Image

from .depth_estimator import bbox_distance
from .detector import Detector
from .sign_localizer import SignLocalizer
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
        self.declare_parameter("rotate_180", True)      # camera mounted upside down
        self.declare_parameter("sign_geometry", False)  # shape-aware localisation
        self.declare_parameter("geometry_config", "")   # "" -> <share>/config/sign_geometry.yaml
        self.declare_parameter("camera_info_topic", "")  # "" -> rgb_topic's camera_info
        self.declare_parameter("geometry_log", "")       # CSV of every estimate ("" = off)

        def param(name):
            return self.get_parameter(name).value

        self.depth_roi_half_size = param("depth_roi_half_size")
        self.show_window = param("show_window")
        self.log_detections = param("log_detections")
        self.rotate_180 = bool(param("rotate_180"))

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

        # --------------------------------------------------------
        # Shape-aware sign localisation (optional)
        # --------------------------------------------------------
        self.localizer = None
        self.geometry_csv = None
        self.geometry_time = 0.0
        self.geometry_count = 0
        if param("sign_geometry"):
            config = param("geometry_config") or os.path.join(
                get_package_share_directory("object_detection"),
                "config", "sign_geometry.yaml")
            self.localizer = SignLocalizer(config, rotated_180=self.rotate_180)
            missing = [n for n in self.detector.class_names.values()
                       if n not in self.localizer.specs]
            if missing:
                self.get_logger().warn(
                    f"No size/shape in {config} for: {', '.join(missing)} "
                    "(stereo only for these)")
            info_topic = param("camera_info_topic") or \
                param("rgb_topic").rsplit("/", 1)[0] + "/camera_info"
            self.camera_info_topic = info_topic
            self.get_logger().info(
                f"Sign geometry ON ({config}); calibration from {info_topic}, "
                "until it arrives from the config file")
            if param("geometry_log"):
                self.geometry_file = open(param("geometry_log"), "w", newline="",
                                          encoding="utf-8")
                self.geometry_csv = csv.writer(self.geometry_file)
                self.geometry_csv.writerow([
                    "stamp", "class", "confidence", "x1", "y1", "x2", "y2",
                    "has_outline", "stereo_box", "stereo_mask", "box_size",
                    "geometry", "geometry_sigma", "view_angle_deg", "method",
                    "fit_error_px", "fused", "fused_sigma", "source", "chi",
                    "consistent", "calibration"])

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

        if self.localizer is not None:
            self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self.camera_info_callback,
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
        if self.rotate_180:
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

        if self.localizer is not None:
            t1 = time.perf_counter()
            for det in detections:
                info = self.localizer.localize(det, frame.shape, depth, det["distance"])
                det["geometry"] = info
                det["distance"] = info["distance"]
                if self.geometry_csv is not None:
                    self.write_geometry_row(msg.header.stamp, det, info)
            self.geometry_time += time.perf_counter() - t1
            self.geometry_count += len(detections)

        self.publish_detections(detections, msg.header)

        if self.log_detections:
            for det in detections:
                dist = (
                    f"{det['distance']:.2f} m"
                    if det["distance"] is not None
                    else "N/A"
                )
                geo = det.get("geometry")
                extra = ""
                if geo is not None:
                    def f(v):
                        return "-" if v is None else f"{v:.2f}"
                    extra = (f" | stereo_box={f(geo['stereo_box'])}"
                             f" stereo_mask={f(geo['stereo_mask'])}"
                             f" geometry={f(geo['geometry'])}"
                             f" box_size={f(geo['box_size'])}"
                             f" angle={f(geo['view_angle_deg'])}"
                             f" -> {geo['source']}"
                             + ("" if geo["consistent"] is not False else " DISAGREE"))
                self.get_logger().info(
                    f"{det['class_name']:20s}"
                    f" Conf={det['confidence']:.2f}"
                    f" Dist={dist}" + extra
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
        self.latest_depth = cv2.rotate(depth, cv2.ROTATE_180) if self.rotate_180 else depth

        if self.show_window:
            depth_vis = cv2.normalize(self.latest_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imshow("Depth", depth_vis)
            cv2.waitKey(1)

    def camera_info_callback(self, msg):
        """Use the camera's own calibration for the geometry (once)."""
        if self.localizer.info_camera is None:
            self.localizer.set_camera_info(list(msg.k), list(msg.d), msg.width, msg.height)
            if self.localizer.info_camera is not None:
                self.get_logger().info(
                    f"Calibration from camera_info: {msg.width}x{msg.height}, "
                    f"fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}, {len(msg.d)} distortion terms")

    def write_geometry_row(self, stamp, det, info):
        """One CSV line per detection with every method's estimate."""
        def v(x):
            return "" if x is None else (f"{x:.4f}" if isinstance(x, float) else x)
        x1, y1, x2, y2 = det["bbox"]
        self.geometry_csv.writerow([
            f"{stamp.sec}.{stamp.nanosec:09d}", det["class_name"], f"{det['confidence']:.3f}",
            x1, y1, x2, y2, int(det.get("polygon") is not None),
            v(info["stereo_box"]), v(info["stereo_mask"]), v(info["box_size"]),
            v(info["geometry"]), v(info["geometry_sigma"]), v(info["view_angle_deg"]),
            v(info["method"]), v(info["fit_error_px"]), v(info["fused"]),
            v(info["fused_sigma"]), v(info["source"]), v(info["chi"]),
            v(info["consistent"]), self.localizer.calibration_source])
        self.geometry_file.flush()

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
            geo = ""
            if self.localizer is not None and self.geometry_count:
                geo = (f", geometry {1000.0 * self.geometry_time / self.geometry_count:.1f}"
                       f" ms/sign ({self.localizer.calibration_source} calibration)")
            self.get_logger().info(
                f"{self.frames} frames, {ms:.1f} ms/frame inference, "
                f"{self.detected} detections, depth: {depth}{geo}")
        else:
            self.get_logger().warn("No frames received - is the camera publishing?")
        self.frames = self.detected = 0
        self.infer_time = 0.0
        self.geometry_time = 0.0
        self.geometry_count = 0


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
