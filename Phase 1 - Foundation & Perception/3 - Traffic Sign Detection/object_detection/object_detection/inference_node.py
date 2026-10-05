#!/usr/bin/env python3
from ament_index_python.packages import get_package_share_directory
import os

from .detector import Detector
from .visualization import Visualizer

import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import qos_profile_sensor_data

from object_detection_msgs.msg import Detection
from object_detection_msgs.msg import DetectionArray

class InferenceNode(Node):

    def __init__(self):

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
        self.declare_parameter("depth_roi_half_size", 3)
        self.declare_parameter("rgb_topic", "/oak/rgb/image_raw")
        self.declare_parameter("depth_topic", "/oak/stereo/image_raw")
        self.declare_parameter("detections_topic", "/detections")

        self.depth_roi_half_size = self.get_parameter("depth_roi_half_size").value
        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        detections_topic = self.get_parameter("detections_topic").value

        # --------------------------------------------------------
        # Load YOLO Model
        # --------------------------------------------------------

        # Empty model_path -> bundled models/last.pt
        model_path = self.get_parameter("model_path").value
        if not model_path:
            model_path = os.path.join(
                get_package_share_directory("object_detection"),
                "models",
                "last.pt"
            )

        self.detector = Detector(
            model_path=model_path,
            confidence=self.get_parameter("confidence").value,
            iou=self.get_parameter("iou").value,
            image_size=self.get_parameter("image_size").value,
            device=self.get_parameter("device").value or None,
        )

        self.visualizer = Visualizer()

        self.subscription = self.create_subscription(
            Image,
            rgb_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.depth_subscription = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )

        self.detection_pub = self.create_publisher(
            DetectionArray,
            detections_topic,
            10,
        )

        self.get_logger().info(f"Subscribed to {rgb_topic}")

    def estimate_distance(self, bbox):

        if self.latest_depth is None:
            return None

        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2


        # (2h+1)x(2h+1) region around the center pixel (7x7 for h=3)
        h = self.depth_roi_half_size
        roi = self.latest_depth[
            max(0, cy - h):min(self.latest_depth.shape[0], cy + h + 1),
            max(0, cx - h):min(self.latest_depth.shape[1], cx + h + 1),
        ]

        # Remove invalid (zero) depth values
        valid = roi[roi > 0]

        if len(valid) == 0:
            return None

        # Median depth in millimetres
        depth_mm = float(np.median(valid))

        self.get_logger().info(f"Raw depth = {depth_mm:.0f} mm")

        # Convert to metres
        return depth_mm / 1000.0

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

        # Camera mounted upside down
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Wait until a depth frame has been received
        if self.latest_depth is None:
            return

        # ---------------------------------------------
        # Run YOLO
        # ---------------------------------------------

        detections = self.detector.detect(frame)

        # Estimate distance for every detection
        for det in detections:
            det["distance"] = self.estimate_distance(det["bbox"])
        
        self.publish_detections(
            detections,
            msg.header,
        )

        # Publish detections
        # self.publish_detections(detections, msg.header)

        # Console output
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
        # Draw detections
        # ---------------------------------------------

        annotated = self.visualizer.draw(
            frame,
            detections
        )

        cv2.imshow("YOLOv8 Detection", annotated)
        cv2.waitKey(1)

    def depth_callback(self, msg):

        self.latest_depth = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="passthrough"
        )

        # Rotate depth exactly like RGB
        self.latest_depth = cv2.rotate(
            self.latest_depth,
            cv2.ROTATE_180
        )

        depth_vis = self.latest_depth.copy()

        depth_vis = cv2.normalize(
            depth_vis,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )

        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        cv2.imshow("Depth", depth_vis)
        cv2.waitKey(1)

    def publish_detections(self, detections, header):

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


def main(args=None):

    rclpy.init(args=args)

    node = InferenceNode()

    rclpy.spin(node)

    node.destroy_node()

    cv2.destroyAllWindows()

    rclpy.shutdown()


if __name__ == "__main__":
    main()