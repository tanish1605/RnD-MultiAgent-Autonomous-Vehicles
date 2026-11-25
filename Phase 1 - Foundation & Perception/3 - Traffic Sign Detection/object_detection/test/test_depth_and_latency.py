"""Unit tests for depth_estimator.py and utils.latency_summary (no ROS needed)."""

import numpy as np
import pytest

from object_detection.depth_estimator import bbox_distance
from object_detection.utils import latency_summary


def test_distance_same_resolution():
    depth = np.zeros((100, 200), dtype=np.uint16)
    depth[45:56, 95:106] = 1500                        # 1.5 m around (100, 50)
    assert bbox_distance(depth, [90, 40, 110, 60], (100, 200)) == pytest.approx(1.5)


def test_distance_scaled_to_smaller_depth_image():
    # colour 1920x1080, depth 640x360 covering the same view: the box centre
    # (960, 540) in colour pixels is (320, 180) in depth pixels
    depth = np.zeros((360, 640), dtype=np.uint16)
    depth[177:184, 317:324] = 2000
    assert bbox_distance(depth, [900, 500, 1020, 580], (1080, 1920)) == pytest.approx(2.0)


def test_distance_ignores_invalid_pixels():
    depth = np.zeros((50, 50), dtype=np.uint16)
    depth[25, 25] = 800                                # one valid pixel in the patch
    assert bbox_distance(depth, [20, 20, 30, 30], (50, 50)) == pytest.approx(0.8)


def test_distance_none_without_valid_depth():
    depth = np.zeros((50, 50), dtype=np.uint16)
    assert bbox_distance(depth, [20, 20, 30, 30], (50, 50)) is None


def test_distance_box_at_image_edge():
    depth = np.full((50, 50), 1000, dtype=np.uint16)
    assert bbox_distance(depth, [45, 45, 60, 60], (50, 50)) == pytest.approx(1.0)


def test_latency_summary():
    lat = [0.010] * 98 + [0.050, 0.100]                # two slow frames
    arr = [i / 30.0 for i in range(100)]               # 30 Hz
    s = latency_summary(lat, arr)
    assert s['count'] == 100
    assert s['rate_hz'] == pytest.approx(30.0)
    assert s['p50_ms'] == pytest.approx(10.0)
    assert s['max_ms'] == pytest.approx(100.0)
    assert s['p99_ms'] > s['p50_ms']


def test_latency_summary_empty():
    assert latency_summary([], []) is None
