"""Unit tests for sign_localizer.py (no ROS needed)."""

import os

import cv2
import numpy as np
import pytest

from object_detection.sign_geometry import load_config, pose_vectors, render_mask
from object_detection.sign_localizer import SignLocalizer

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'sign_geometry.yaml')


def detection(name, spec, camera, z, yaw_deg=0.0, x=0.0):
    """Render a sign at depth z; return its detection dict (box + outline) and mask."""
    rvec, tvec = pose_vectors(z, yaw_deg=yaw_deg, x=x)
    mask = render_mask(spec, camera, rvec, tvec)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    poly = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    return {'class_name': name, 'confidence': 0.9,
            'bbox': [int(x1), int(y1), int(x2), int(y2)], 'polygon': poly}, mask


@pytest.mark.parametrize('name, z', [('stop', 0.6), ('one_way', 1.0),
                                     ('crossroads', 0.8), ('priority_road', 0.7)])
def test_geometry_finds_the_distance(name, z):
    loc = SignLocalizer(CONFIG, rotated_180=True)
    camera = loc.camera_for(1920, 1080)
    det, _ = detection(name, loc.specs[name], camera, z, yaw_deg=20)
    info = loc.localize(det, (1080, 1920, 3))
    assert info['geometry'] == pytest.approx(z, rel=0.05)
    assert info['source'] == 'geometry' and info['distance'] == info['geometry']
    assert info['box_size'] is not None


def test_fused_with_stereo_and_flags_disagreement():
    loc = SignLocalizer(CONFIG, rotated_180=True)
    camera = loc.camera_for(1920, 1080)
    det, mask = detection('stop', loc.specs['stop'], camera, 0.6)
    depth = np.zeros((1080, 1920), np.uint16)
    depth[mask > 0] = 600                                   # stereo agrees: 0.60 m
    info = loc.localize(det, (1080, 1920, 3), depth, stereo_box=0.61)
    assert info['stereo_mask'] == pytest.approx(0.6)
    assert info['source'] == 'fused' and info['consistent']
    assert info['fused'] == pytest.approx(0.6, rel=0.03)
    depth[mask > 0] = 1500                                  # stereo says 1.5 m: a poster?
    info = loc.localize(det, (1080, 1920, 3), depth)
    assert info['consistent'] is False


def test_box_model_without_outline_falls_back_to_stereo():
    loc = SignLocalizer(CONFIG)
    det = {'class_name': 'stop', 'confidence': 0.9, 'bbox': [900, 500, 1000, 600]}
    info = loc.localize(det, (1080, 1920, 3), None, stereo_box=0.57)
    assert info['geometry'] is None and info['source'] == 'stereo'
    assert info['distance'] == pytest.approx(0.57)
    assert loc.localize({'class_name': 'unknown_sign', 'bbox': [0, 0, 5, 5]},
                        (1080, 1920, 3))['distance'] is None


def test_calibration_scaled_and_from_camera_info():
    loc = SignLocalizer(CONFIG, rotated_180=False)
    base, _, _ = load_config(CONFIG)
    small = loc.camera_for(960, 540)
    assert small.K[0, 0] == pytest.approx(base.K[0, 0] / 2)
    assert loc.calibration_source == 'config'
    loc.set_camera_info([1000, 0, 640, 0, 1000, 360, 0, 0, 1], [0.0] * 8, 1280, 720)
    assert loc.calibration_source == 'camera_info'
    assert loc.camera_for(1280, 720).K[0, 0] == 1000
    loc.set_camera_info([0] * 9, [], 0, 0)                  # empty info is ignored
    assert loc.calibration_source == 'camera_info'
