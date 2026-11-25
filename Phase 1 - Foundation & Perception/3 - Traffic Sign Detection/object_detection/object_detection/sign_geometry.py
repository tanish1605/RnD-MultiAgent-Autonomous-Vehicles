#!/usr/bin/env python3

"""
sign_geometry.py.

Metric pose of a traffic sign from its segmentation outline.

A traffic sign has a known shape and size. From its outline (an instance
segmentation mask) we fit that shape - an ellipse for a round sign, corners
for a triangle, square, diamond, rectangle or octagon - and, with the camera
calibration, recover where the sign is in 3D and the angle it is seen at.

This gives a distance that does not use the stereo depth at all. The two are
complementary: stereo is very accurate close up but its error grows with the
square of the distance, while the size-based estimate degrades only with the
distance itself. fuse_distance() combines them by their uncertainties and
flags detections where they disagree (a poster of a sign, a reflection, a
wrong class).

Coordinate conventions
----------------------
Camera frame : x right, y down, z forward (OpenCV). "z" is the depth along the
               optical axis, which is what the OAK-D depth image stores.
Sign frame   : centred on the sign, x right, y down, in the sign's plane;
               z points away from a viewer the sign faces.

No ROS code should exist here.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np
import yaml

# Number of corners per polygonal shape (a circle has none)
CORNERS = {
    'triangle_up': 3,
    'triangle_down': 3,
    'square': 4,
    'rectangle': 4,
    'diamond': 4,
    'octagon': 8,
}
SHAPES = ('circle',) + tuple(CORNERS)

_COS_22_5 = math.cos(math.radians(22.5))


# ── data types ──────────────────────────────────────────────────────────────

@dataclass
class SignSpec:
    """
    Physical description of one sign class.

    shape     : one of SHAPES
    width     : metres; circle = diameter, triangle / square = side length,
                diamond = corner-to-corner width, octagon = width across flats
    height    : metres, rectangle only (defaults to width)
    size_rel_sigma : relative uncertainty of the size (0.02 for a sign measured
                with a ruler, larger for a guessed size)
    """

    shape: str
    width: float
    height: float = None
    size_rel_sigma: float = 0.02

    def __post_init__(self):
        if self.shape not in SHAPES:
            raise ValueError(f'Unknown shape {self.shape!r}; use one of {SHAPES}')
        if self.height is None:
            self.height = self.width

    def vertical_extent(self):
        """Top-to-bottom size of the upright sign (used by the box baseline)."""
        if self.shape in ('triangle_up', 'triangle_down'):
            return self.width * math.sqrt(3) / 2
        return self.height


@dataclass
class Camera:
    """Pinhole camera with OpenCV distortion (up to 14 coefficients)."""

    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).ravel()
        self.K_inv = np.linalg.inv(self.K)

    def rotated_180(self):
        """
        Calibration of the same camera after the image is rotated by 180 deg.

        The OAK-D is mounted upside down and every node rotates the image
        before processing, so geometry on the rotated image needs this.
        A 180 deg rotation maps normalised coordinates (x, y) -> (-x, -y):
        the principal point mirrors, and the distortion terms that are even
        in (x, y) - tangential p1, p2, thin-prism s1..s4 and the sensor tilt
        tau_x, tau_y - change sign. Radial terms are unchanged.
        """
        K = self.K.copy()
        K[0, 2] = (self.width - 1) - K[0, 2]
        K[1, 2] = (self.height - 1) - K[1, 2]
        d = np.zeros(14)
        d[:len(self.dist)] = self.dist
        for i in (2, 3, 8, 9, 10, 11, 12, 13):      # p1 p2 s1 s2 s3 s4 tx ty
            d[i] = -d[i]
        return Camera(K, d[:len(self.dist)], self.width, self.height)

    def undistort_px(self, points_px):
        """Distorted pixel coordinates -> ideal (distortion-free) pixels."""
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        if not np.any(self.dist):
            return pts.reshape(-1, 2)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)
        if hasattr(cv2, 'undistortPointsIter'):        # OpenCV 4.x
            out = cv2.undistortPointsIter(pts, self.K, self.dist, R=None, P=self.K,
                                          criteria=criteria)
        else:                                           # OpenCV 5.x: criteria moved here
            out = cv2.undistortPoints(pts, self.K, self.dist, R=None, P=self.K,
                                      criteria=criteria)
        return out.reshape(-1, 2)

    def normalised(self, ideal_px):
        """Ideal pixels -> normalised image coordinates (x/z, y/z)."""
        p = np.asarray(ideal_px, dtype=np.float64).reshape(-1, 2)
        homog = np.hstack([p, np.ones((len(p), 1))]) @ self.K_inv.T
        return homog[:, :2]


@dataclass
class SignPose:
    """Result of estimate_pose()."""

    shape: str
    method: str              # 'ellipse' or 'pnp'
    z: float                 # depth along the optical axis (m), comparable to stereo depth
    range: float             # straight-line distance camera -> sign centre (m)
    position: np.ndarray     # sign centre in the camera frame (m)
    view_angle_deg: float    # angle between the line of sight and the sign's normal
    sigma_z: float           # 1-sigma uncertainty of z (m)
    fit_error_px: float      # rms distance between the outline and the fitted shape
    size_px: float           # apparent size of the sign in pixels


@dataclass
class FusedDistance:
    """Result of fuse_distance()."""

    z: float
    sigma: float
    source: str              # 'fused', 'geometry' or 'stereo'
    consistent: bool = None  # None when only one estimate was available
    chi: float = None        # disagreement in units of the combined sigma


# ── loading ─────────────────────────────────────────────────────────────────

def load_config(path):
    """
    Read config/sign_geometry.yaml.

    Returns (camera, {class_name: SignSpec}, stereo_params).
    """
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cam = cfg['camera']
    camera = Camera(cam['k'], cam['d'], cam['width'], cam['height'])
    if cam.get('image_rotated_180', False):
        camera = camera.rotated_180()
    guessed_sigma = float(cfg.get('guessed_size_rel_sigma', 0.25))
    specs = {}
    for name, s in cfg['signs'].items():
        specs[name] = SignSpec(
            shape=s['shape'],
            width=float(s['width_m']),
            height=float(s['height_m']) if 'height_m' in s else None,
            size_rel_sigma=0.02 if s.get('measured', False) else guessed_sigma,
        )
    return camera, specs, cfg.get('stereo', {})


# ── sign model ──────────────────────────────────────────────────────────────

def model_outline(spec, n_circle=64):
    """
    Outline of the sign in its own plane: (N, 3) points in metres, z = 0.

    Polygon corners are returned clockwise as seen by a viewer the sign faces
    (x right, y down); a circle is sampled with n_circle points.
    """
    w, h = spec.width, spec.height
    if spec.shape == 'circle':
        a = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
        xy = np.stack([w / 2 * np.cos(a), w / 2 * np.sin(a)], axis=1)
    elif spec.shape in ('square', 'rectangle'):
        xy = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    elif spec.shape == 'diamond':
        xy = np.array([[0, -w / 2], [w / 2, 0], [0, w / 2], [-w / 2, 0]])
    elif spec.shape == 'triangle_up':
        t = w * math.sqrt(3) / 2                          # height of the triangle
        xy = np.array([[0, -2 * t / 3], [w / 2, t / 3], [-w / 2, t / 3]])
    elif spec.shape == 'triangle_down':
        t = w * math.sqrt(3) / 2
        xy = np.array([[-w / 2, -t / 3], [w / 2, -t / 3], [0, 2 * t / 3]])
    else:                                                 # octagon, flat top
        r = w / 2 / _COS_22_5                             # circumradius
        a = np.radians(22.5 + 45 * np.arange(8))
        xy = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    return np.hstack([xy, np.zeros((len(xy), 1))])


# ── pose estimation ─────────────────────────────────────────────────────────

def estimate_pose(mask, spec, camera, edge_noise_px=1.0, min_area_px=40):
    """
    Pose of one sign from its binary mask (HxW, nonzero = sign).

    The mask must have the camera's resolution. Returns a SignPose, or None
    when the mask is too small or the shape cannot be fitted.
    """
    contour = _largest_contour(mask)
    if contour is None or cv2.contourArea(contour) < min_area_px:
        return None
    ideal = camera.undistort_px(contour.reshape(-1, 2))
    area_px = cv2.contourArea(ideal.astype(np.float32))

    if spec.shape == 'circle':
        return _ellipse_pose(ideal, spec, camera, spec.width, edge_noise_px)

    pose = _polygon_pose(ideal, spec, camera, area_px, edge_noise_px)
    if pose is None and spec.shape == 'octagon':
        # small octagons look round: fall back to an ellipse of equal size
        d_eq = spec.width * (1 + 1 / _COS_22_5) / 2
        pose = _ellipse_pose(ideal, spec, camera, d_eq, edge_noise_px)
    return pose


def _largest_contour(mask):
    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _sigma_z(z, spec, size_px, fit_error_px, edge_noise_px):
    rel_px = max(edge_noise_px, fit_error_px) / max(size_px, 1.0)
    return z * math.hypot(rel_px, spec.size_rel_sigma)


def _ellipse_pose(ideal, spec, camera, diameter, edge_noise_px):
    """
    Round sign: fit an ellipse to the outline.

    A circle turned away from the camera shrinks only across the turning
    axis, so the ellipse's major axis still spans the full diameter:
    z = diameter / (major axis length in normalised coordinates), and the
    axis ratio gives the viewing angle: cos(angle) = minor / major.
    """
    if len(ideal) < 5:
        return None
    (cu, cv), (aw, ah), ang = cv2.fitEllipse(ideal.astype(np.float32))
    major, minor = (aw, ah) if aw >= ah else (ah, aw)
    theta = math.radians(ang if aw >= ah else ang + 90.0)
    u = np.array([math.cos(theta), math.sin(theta)])
    v = np.array([-u[1], u[0]])
    c = np.array([cu, cv])
    ends = np.array([c + u * major / 2, c - u * major / 2,
                     c + v * minor / 2, c - v * minor / 2])
    n = camera.normalised(ends)
    len_major = np.linalg.norm(n[0] - n[1])
    len_minor = np.linalg.norm(n[2] - n[3])
    if len_major <= 0:
        return None

    z = diameter / len_major
    centre = camera.normalised([c])[0]
    position = z * np.array([centre[0], centre[1], 1.0])
    view = math.degrees(math.acos(min(1.0, len_minor / len_major)))

    # rms radial distance of the outline from the fitted ellipse, in pixels
    rel = ideal - c
    x = rel @ u / (major / 2)
    y = rel @ v / (minor / 2)
    r = np.sqrt(x ** 2 + y ** 2)
    fit_error = float(np.sqrt(np.mean((r - 1) ** 2)) * (major + minor) / 4)

    return SignPose(spec.shape, 'ellipse', float(z), float(np.linalg.norm(position)),
                    position, view, _sigma_z(z, spec, major, fit_error, edge_noise_px),
                    fit_error, float(major))


def _polygon_corners(ideal, n):
    """Exactly n corners of the outline's convex hull, or None."""
    hull = cv2.convexHull(ideal.astype(np.float32))
    if len(hull) < n:
        return None
    perimeter = cv2.arcLength(hull, True)
    lo, hi = 0.0, 0.25 * perimeter
    for _ in range(40):                     # binary search for the right tolerance
        eps = (lo + hi) / 2
        approx = cv2.approxPolyDP(hull, eps, True)
        if len(approx) == n:
            return approx.reshape(-1, 2).astype(np.float64)
        if len(approx) > n:
            lo = eps
        else:
            hi = eps
    return None


def _clockwise(points):
    """Sort points clockwise on screen (image y axis points down)."""
    c = points.mean(axis=0)
    order = np.argsort(np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0]))
    return points[order]


def _upright_pose(ideal, corners, spec, camera, area_px, edge_noise_px):
    """
    Triangle: pose from its height, assuming the sign stands upright.

    Three corners leave the full 3D pose ambiguous (several poses project to
    the same triangle) and very sensitive to corner noise on small signs. An
    upright sign turned about its vertical axis gets narrower but not
    shorter, so the vertical extent gives the depth and the width/height
    ratio gives the viewing angle - the same idea as for a round sign.
    """
    n = camera.normalised(corners)
    height_n = n[:, 1].max() - n[:, 1].min()
    width_n = n[:, 0].max() - n[:, 0].min()
    if height_n <= 0:
        return None
    z = spec.vertical_extent() / height_n
    view = math.degrees(math.acos(min(1.0, width_n * z / spec.width)))
    centre = n.mean(axis=0)                  # the centroid survives the projection
    position = z * np.array([centre[0], centre[1], 1.0])

    poly = corners.astype(np.float32).reshape(-1, 1, 2)
    dist = [cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in ideal]
    fit_error = float(np.sqrt(np.mean(np.square(dist))))
    size_px = float(corners[:, 1].max() - corners[:, 1].min())
    return SignPose(spec.shape, 'upright', float(z), float(np.linalg.norm(position)),
                    position, view, _sigma_z(z, spec, size_px, fit_error, edge_noise_px),
                    fit_error, size_px)


def _polygon_pose(ideal, spec, camera, area_px, edge_noise_px):
    """
    Polygonal sign: find its corners and solve the perspective-n-point problem.

    Triangles go to _upright_pose(). For four or more corners, the corners of
    the image and of the model are both ordered clockwise and every cyclic
    pairing is tried. Poses behind the camera or showing the sign's back are
    dropped. A planar target can have two poses that fit almost equally well;
    among those within a small margin of the best reprojection error the most
    upright one is chosen, since signs stand vertically.
    """
    n = CORNERS[spec.shape]
    corners = _polygon_corners(ideal, n)
    if corners is None:
        return None
    if n == 3:
        return _upright_pose(ideal, corners, spec, camera, area_px, edge_noise_px)
    img = _clockwise(corners)
    model = model_outline(spec)
    model = model[np.argsort(np.arctan2(model[:, 1], model[:, 0]))]

    candidates = []
    for shift in range(n):
        obj = np.roll(model, shift, axis=0)
        ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, img.reshape(-1, 1, 2), camera.K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        for rvec, tvec, err in zip(rvecs, tvecs, np.ravel(errs)):
            R, t = cv2.Rodrigues(rvec)[0], tvec.ravel()
            if t[2] <= 0 or R[:, 2] @ t <= 0:    # behind the camera or seen from the back
                continue
            tilt = math.acos(max(-1.0, min(1.0, R[1, 1])))   # sign's y vs camera's y
            candidates.append((float(err), tilt, t, R[:, 2]))
    if not candidates:
        return None

    best_err = min(c[0] for c in candidates)
    close = [c for c in candidates if c[0] <= 1.5 * best_err + 0.5]
    err, _, t, normal = min(close, key=lambda c: c[1])
    rng = float(np.linalg.norm(t))
    cos_view = abs(normal @ t) / (np.linalg.norm(normal) * rng)
    view = math.degrees(math.acos(min(1.0, cos_view)))
    size_px = math.sqrt(max(area_px, 1.0))
    return SignPose(spec.shape, 'pnp', float(t[2]), rng, t, view,
                    _sigma_z(t[2], spec, size_px, err, edge_noise_px), err, size_px)


# ── other distance estimates and fusion ─────────────────────────────────────

def box_size_distance(bbox, spec, camera):
    """
    Baseline: depth from the height of the detection box alone.

    z = fy * real height / box height. Uses the vertical size because turning
    an upright sign about its vertical axis does not shrink it.
    """
    x1, y1, x2, y2 = bbox
    h_px = max(y2 - y1, 1)
    return camera.K[1, 1] * spec.vertical_extent() / h_px


def stereo_depth_in_mask(depth_mm, mask, erode_px=2):
    """
    Median stereo depth (m) of the pixels inside the sign's mask, or None.

    The mask is scaled to the depth image if their sizes differ (depth
    aligned to the colour camera is assumed) and eroded so that pixels on
    the sign's edge, which often carry background depth, are left out.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    dh, dw = depth_mm.shape[:2]
    if m.shape[:2] != (dh, dw):
        m = cv2.resize(m, (dw, dh), interpolation=cv2.INTER_NEAREST)
    if erode_px > 0:
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        eroded = cv2.erode(m, k)
        if eroded.any():
            m = eroded
    values = depth_mm[m > 0]
    values = values[values > 0]
    if values.size == 0:
        return None
    return float(np.median(values)) / 1000.0


def stereo_sigma(z, focal_px, baseline_m, disparity_sigma_px=0.25, floor_m=0.005):
    """
    1-sigma error of stereo depth at depth z.

    From z = f * B / disparity: dz = z^2 / (f * B) * d(disparity), so the
    error grows with the square of the distance.
    """
    return max(floor_m, z * z * disparity_sigma_px / (focal_px * baseline_m))


def fuse_distance(geometry=None, stereo=None, gate=3.0):
    """
    Combine (z, sigma) estimates by inverse-variance weighting.

    geometry, stereo : (z, sigma) tuples or None
    gate             : estimates further apart than gate x combined sigma
                       are flagged as inconsistent (consistent=False); the
                       detection may then be wrong (poster, reflection,
                       wrong class) and should be trusted less.
    Returns a FusedDistance, or None when neither estimate is available.
    """
    if geometry is None and stereo is None:
        return None
    if geometry is None:
        return FusedDistance(stereo[0], stereo[1], 'stereo')
    if stereo is None:
        return FusedDistance(geometry[0], geometry[1], 'geometry')

    (zg, sg), (zs, ss) = geometry, stereo
    chi = abs(zg - zs) / math.hypot(sg, ss)
    wg, ws = 1.0 / sg ** 2, 1.0 / ss ** 2
    z = (wg * zg + ws * zs) / (wg + ws)
    return FusedDistance(z, math.sqrt(1.0 / (wg + ws)), 'fused', chi <= gate, chi)


# ── simulation helpers (tests and accuracy studies) ─────────────────────────

def render_mask(spec, camera, rvec, tvec, n_circle=256):
    """Binary mask (camera resolution) of the sign at pose (rvec, tvec)."""
    pts, _ = cv2.projectPoints(model_outline(spec, n_circle),
                               np.asarray(rvec, np.float64), np.asarray(tvec, np.float64),
                               camera.K, camera.dist)
    mask = np.zeros((camera.height, camera.width), np.uint8)
    cv2.fillPoly(mask, [np.round(pts.reshape(-1, 2)).astype(np.int32)], 1)
    return mask


def pose_vectors(z, yaw_deg=0.0, x=0.0, y=0.0, roll_deg=0.0):
    """(rvec, tvec) of a sign at (x, y, z) turned by yaw about its vertical axis."""
    yaw, roll = math.radians(yaw_deg), math.radians(roll_deg)
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]])
    rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]])
    rvec = cv2.Rodrigues(ry @ rz)[0]
    return rvec, np.array([[x], [y], [z]], np.float64)
