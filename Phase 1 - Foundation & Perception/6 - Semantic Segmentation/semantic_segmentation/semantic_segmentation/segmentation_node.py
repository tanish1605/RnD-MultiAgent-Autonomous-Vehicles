#!/usr/bin/env python3
"""
segmentation_node.py.

ROS 2 node - node name: segmentation_node

Subscribes  (topic names are parameters, see config/segmentation_params.yaml)
----------
/oak/rgb/image_raw        sensor_msgs/Image   raw OAK-D camera feed

Publishes
---------
/segmentation/mask        sensor_msgs/Image   mono8, pixel value = class index
/segmentation/overlay     sensor_msgs/Image   bgr8 colour overlay (debug); only
                                              computed while something subscribes
/segmentation/classes     std_msgs/String     JSON list of class names, latched
                                              (transient local) - index = mask value

The mask is in the UPRIGHT orientation (rotated 180 deg like every other node)
and, by default, at the camera's resolution, so it lines up pixel-for-pixel
with what lane_detection works on.
"""

import json
import os
import time

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSDurabilityPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .segmenter import Segmenter


class SegmentationNode(Node):
    """Segment every camera frame and publish class-index masks."""

    def __init__(self):
        """Declare parameters, load the model, create publishers/subscribers."""
        super().__init__('segmentation_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'auto')
        self.declare_parameter('rotate_180', True)
        self.declare_parameter('full_resolution_mask', True)
        self.declare_parameter('process_every_n', 1)
        self.declare_parameter('overlay_alpha', 0.5)
        self.declare_parameter('stats_period_sec', 5.0)
        self.declare_parameter('image_topic', '/oak/rgb/image_raw')
        self.declare_parameter('mask_topic', '/segmentation/mask')
        self.declare_parameter('overlay_topic', '/segmentation/overlay')
        self.declare_parameter('classes_topic', '/segmentation/classes')

        def param(name):
            return self.get_parameter(name).value

        self.rotate = param('rotate_180')
        self.full_res = param('full_resolution_mask')
        self.every_n = max(1, int(param('process_every_n')))
        self.alpha = float(param('overlay_alpha'))

        # Empty model_path -> bundled models/seg.onnx
        model_path = param('model_path') or os.path.join(
            get_package_share_directory('semantic_segmentation'), 'models', 'seg.onnx')
        self.segmenter = Segmenter(model_path, device=param('device'))
        self.get_logger().info(
            f'Model {model_path} | backend {self.segmenter.backend} | '
            f'input {self.segmenter.input_w}x{self.segmenter.input_h} | '
            f'classes {self.segmenter.class_names}')

        self.bridge = CvBridge()
        self.mask_pub = self.create_publisher(Image, param('mask_topic'), 1)
        self.overlay_pub = self.create_publisher(Image, param('overlay_topic'), 1)

        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.classes_pub = self.create_publisher(String, param('classes_topic'), latched)
        self.classes_pub.publish(String(data=json.dumps(self.segmenter.class_names)))

        # sensor-data QoS (best effort, depth 5): if inference is slower than
        # the camera, old frames are dropped instead of piling up.
        self.create_subscription(Image, param('image_topic'), self.image_callback,
                                 qos_profile_sensor_data)

        self.frame_count = 0
        self.processed = 0
        self.infer_time = 0.0
        self.create_timer(float(param('stats_period_sec')), self.log_stats)
        self.get_logger().info(f'Subscribed to {param("image_topic")}')

    def image_callback(self, msg):
        """Segment one frame and publish the mask (+ overlay if wanted)."""
        self.frame_count += 1
        if (self.frame_count - 1) % self.every_n:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # noqa: BLE001 - bad frames must not kill the node
            self.get_logger().error(f'cv_bridge failed: {exc}')
            return
        if self.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)   # camera mounted upside down

        t0 = time.perf_counter()
        mask = self.segmenter.segment(frame, full_resolution=self.full_res)
        self.infer_time += time.perf_counter() - t0
        self.processed += 1

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        # The overlay costs CPU - only build it when someone is watching.
        if self.overlay_pub.get_subscription_count() > 0:
            vis = self.segmenter.overlay(frame, mask, self.alpha)
            vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            vis_msg.header = msg.header
            self.overlay_pub.publish(vis_msg)

    def log_stats(self):
        """Log throughput periodically instead of once per frame."""
        if self.processed:
            ms = 1000.0 * self.infer_time / self.processed
            self.get_logger().info(
                f'{self.processed} frames segmented, {ms:.1f} ms/frame '
                f'(received {self.frame_count})')
        else:
            self.get_logger().warn('No frames segmented yet - is the camera publishing?')
        self.frame_count = self.processed = 0
        self.infer_time = 0.0


def main(args=None):
    """Entry point for `ros2 run semantic_segmentation segmentation_node`."""
    rclpy.init(args=args)
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
