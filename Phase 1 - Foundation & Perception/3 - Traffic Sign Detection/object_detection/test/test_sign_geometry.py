"""Unit tests for sign_geometry.py (no ROS needed)."""

import math
import os

import cv2
import numpy as np
import pytest

from object_detection.sign_geometry import (
    box_size_distance, Camera, estimate_pose, fuse_distance, load_config,
    model_outline, pose_vectors, render_mask, SHAPES, SignSpec,
    stereo_depth_in_mask, stereo_sigma)

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'sign_geometry.yaml')


@pytest.fixture(scope='module')
def camera():
    cam, _, _ = load_config(CONFIG)
    return cam


def spec_for(shape):
    return SignSpec(shape, 0.10, 0.12 if shape == 'rectangle' else None)


# ── model and camera ────────────────────────────────────────────────────────

def test_model_outline_sizes():
    tri = model_outline(SignSpec('triangle_up', 0.1))[:, :2]
    sides = np.linalg.norm(tri - np.roll(tri, 1, axis=0), axis=1)
    assert np.allclose(sides, 0.1)
    assert np.allclose(tri.mean(axis=0), 0, atol=1e-12)          # centred
    octa = model_outline(SignSpec('octagon', 0.1))[:, :2]
    assert octa[:, 1].max() - octa[:, 1].min() == pytest.approx(0.1)   # across flats
    circ = model_outline(SignSpec('circle', 0.1))[:, :2]
    assert np.allclose(np.linalg.norm(circ, axis=1), 0.05)


def test_config_loads_every_class():
    cam, specs, stereo = load_config(CONFIG)
    assert len(specs) == 30
    assert all(s.shape in SHAPES for s in specs.values())
    assert stereo['baseline_m'] > 0
    assert (cam.width, cam.height) == (1920, 1080)


def test_rotated_180_matches_rotated_image():
    """Projecting with the rotated calibration = rotating the projected pixels."""
    cam, _, _ = load_config(CONFIG)
    raw = Camera(cam.K, cam.dist, cam.width, cam.height).rotated_180()  # back to unrotated
    rot = raw.rotated_180()
    pts = np.array([[0.1, -0.05, 1.0], [-0.2, 0.1, 2.0], [0.05, 0.08, 0.7]])
    zero = np.zeros(3)
    p_raw, _ = cv2.projectPoints(pts, zero, zero, raw.K, raw.dist)
    # rotating the image by 180 deg = rotating the scene by 180 deg about the optical axis
    flipped = pts * np.array([-1, -1, 1])
    p_rot, _ = cv2.projectPoints(flipped, zero, zero, rot.K, rot.dist)
    expected = np.array([raw.width - 1, raw.height - 1]) - p_raw.reshape(-1, 2)
    assert np.allclose(p_rot.reshape(-1, 2), expected, atol=1e-6)


def test_undistort_round_trip(camera):
    """The strong OAK-D distortion model is inverted to well below a pixel."""
    pts = np.array([[0.0, 0.0, 1.0], [0.3, 0.15, 1.0], [-0.35, -0.2, 1.0]])
    zero = np.zeros(3)
    distorted, _ = cv2.projectPoints(pts, zero, zero, camera.K, camera.dist)
    ideal, _ = cv2.projectPoints(pts, zero, zero, camera.K, None)
    back = camera.undistort_px(distorted.reshape(-1, 2))
    assert np.abs(back - ideal.reshape(-1, 2)).max() < 0.05


# ── pose ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('z', [0.5, 1.5, 3.0])
def test_distance_frontal(camera, shape, z):
    spec = spec_for(shape)
    pose = estimate_pose(render_mask(spec, camera, *pose_vectors(z, y=0.05)), spec, camera)
    assert pose is not None
    assert pose.z == pytest.approx(z, rel=0.02)


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('z,yaw,x', [(1.5, 30, 0.0), (1.0, -35, 0.2), (3.0, 30, -0.3)])
def test_distance_and_angle_turned(camera, shape, z, yaw, x):
    spec = spec_for(shape)
    rvec, tvec = pose_vectors(z, yaw, x=x, y=0.05)
    pose = estimate_pose(render_mask(spec, camera, rvec, tvec), spec, camera)
    assert pose is not None
    assert pose.z == pytest.approx(z, rel=0.03)
    # true angle between the sign's normal and the line of sight
    normal = cv2.Rodrigues(rvec)[0][:, 2]
    t = tvec.ravel()
    truth = math.degrees(math.acos(abs(normal @ t) / np.linalg.norm(t)))
    assert pose.view_angle_deg == pytest.approx(truth, abs=5)


def test_position_off_centre(camera):
    spec = spec_for('circle')
    pose = estimate_pose(render_mask(spec, camera, *pose_vectors(2.0, x=0.4, y=-0.1)),
                         spec, camera)
    assert np.allclose(pose.position, [0.4, -0.1, 2.0], atol=0.03)
    assert pose.range == pytest.approx(math.sqrt(0.4 ** 2 + 0.1 ** 2 + 2.0 ** 2), rel=0.02)


def test_uncertainty_grows_with_distance(camera):
    spec = SignSpec('circle', 0.10, size_rel_sigma=0.0)
    near = estimate_pose(render_mask(spec, camera, *pose_vectors(0.6)), spec, camera)
    far = estimate_pose(render_mask(spec, camera, *pose_vectors(3.0)), spec, camera)
    assert far.sigma_z / far.z > near.sigma_z / near.z
    guessed = SignSpec('circle', 0.10, size_rel_sigma=0.25)
    pose = estimate_pose(render_mask(guessed, camera, *pose_vectors(1.0)), guessed, camera)
    assert pose.sigma_z >= 0.25 * pose.z


def test_tiny_or_empty_mask(camera):
    spec = spec_for('square')
    assert estimate_pose(np.zeros((camera.height, camera.width), np.uint8), spec, camera) is None
    tiny = np.zeros((camera.height, camera.width), np.uint8)
    tiny[500:503, 900:903] = 1
    assert estimate_pose(tiny, spec, camera) is None


# ── other estimates and fusion ──────────────────────────────────────────────

def test_box_size_distance(camera):
    spec = spec_for('circle')
    mask = render_mask(spec, camera, *pose_vectors(2.0))
    x, y, w, h = cv2.boundingRect(mask)
    assert box_size_distance([x, y, x + w, y + h], spec, camera) == pytest.approx(2.0, rel=0.05)


def test_stereo_depth_in_mask_scales_to_depth_resolution():
    mask = np.zeros((1080, 1920), np.uint8)
    mask[400:600, 800:1000] = 1
    depth = np.zeros((360, 640), np.uint16)
    depth[133:200, 267:333] = 1800                 # the sign, at 1/3 resolution
    depth[133, 267:333] = 5000                     # background bleeding onto the edge
    assert stereo_depth_in_mask(depth, mask) == pytest.approx(1.8)
    assert stereo_depth_in_mask(np.zeros((360, 640), np.uint16), mask) is None


def test_stereo_sigma_grows_with_square_of_distance():
    s1 = stereo_sigma(1.0, 800, 0.075, floor_m=0)
    s3 = stereo_sigma(3.0, 800, 0.075, floor_m=0)
    assert s3 == pytest.approx(9 * s1)


def test_fusion_weights_and_consistency():
    near = fuse_distance(geometry=(1.05, 0.05), stereo=(1.00, 0.005))
    assert near.source == 'fused' and near.z == pytest.approx(1.0, abs=0.002)
    assert near.consistent
    far = fuse_distance(geometry=(3.0, 0.06), stereo=(3.3, 0.3))
    assert abs(far.z - 3.0) < abs(far.z - 3.3)       # leans on geometry far away
    assert far.sigma < 0.06
    poster = fuse_distance(geometry=(1.0, 0.02), stereo=(2.5, 0.03))
    assert poster.consistent is False and poster.chi > 3
    only = fuse_distance(geometry=None, stereo=(2.0, 0.1))
    assert only.source == 'stereo' and only.consistent is None
    assert fuse_distance() is None
