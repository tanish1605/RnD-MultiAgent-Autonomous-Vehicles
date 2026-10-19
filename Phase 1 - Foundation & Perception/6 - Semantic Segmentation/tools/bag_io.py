"""
bag_io.py
=========
Read ROS 2 bags (.db3 or .mcap) WITHOUT a ROS installation, using the
pure-Python `rosbags` library (pip install rosbags). Works on a laptop
and on the Jetson alike.

Used by lane_failure_report.py and extract_frames.py.
"""

from pathlib import Path

import cv2
import numpy as np


def _open_reader(bag_path):
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise SystemExit(
            'Reading bags needs the `rosbags` package:  pip install rosbags'
        ) from exc
    # Humble .db3 bags do not embed message definitions -> supply them.
    return AnyReader([Path(bag_path)],
                     default_typestore=get_typestore(Stores.ROS2_HUMBLE))


def list_topics(bag_path):
    """Return {topic: msgtype} for every topic in the bag."""
    with _open_reader(bag_path) as reader:
        return {c.topic: c.msgtype for c in reader.connections}


def iter_messages(bag_path, topic):
    """Yield (timestamp_ns, deserialized_msg) for one topic, in time order."""
    with _open_reader(bag_path) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            available = ', '.join(sorted(c.topic for c in reader.connections))
            raise SystemExit(f'Topic {topic} not in bag. Available: {available}')
        for conn, t_ns, raw in reader.messages(connections=conns):
            yield t_ns, reader.deserialize(raw, conn.msgtype)


def image_msg_to_bgr(msg):
    """Convert a sensor_msgs/Image or sensor_msgs/CompressedImage to BGR."""
    if hasattr(msg, 'format'):                      # CompressedImage
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    enc = msg.encoding.lower()
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ('bgr8', 'rgb8'):
        img = data.reshape(msg.height, msg.step)[:, :msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3)
        return img if enc == 'bgr8' else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc in ('bgra8', 'rgba8'):
        img = data.reshape(msg.height, msg.step)[:, :msg.width * 4]
        img = img.reshape(msg.height, msg.width, 4)
        code = cv2.COLOR_BGRA2BGR if enc == 'bgra8' else cv2.COLOR_RGBA2BGR
        return cv2.cvtColor(img, code)
    if enc == 'mono8':
        img = data.reshape(msg.height, msg.step)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc in ('yuv422', 'yuv422_yuy2', 'yuyv'):
        img = data.reshape(msg.height, msg.step)[:, :msg.width * 2]
        img = img.reshape(msg.height, msg.width, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    if enc in ('nv12',):
        img = data[:msg.height * msg.width * 3 // 2].reshape(msg.height * 3 // 2, msg.width)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)
    raise ValueError(f'Unsupported image encoding: {msg.encoding}')
