#!/usr/bin/env python3

"""
label_audit.py.

Find likely label errors in a polygon (instance segmentation) sign dataset
by checking every outline against the shape its class should have.

Traffic signs come in a few shapes - round, triangle pointing up or down,
square (or rectangle) and diamond - and every sign class has one of them.
A polygon whose shape does not match the usual shape of its class is either
a wrong class or a badly drawn outline. The expected shape of each class is
learned from the dataset itself (the majority shape of the class), so no
list of class names or shapes is needed.

The shape test uses features that survive the image being stretched (many
exports resize to a square image): which simple shape - an ellipse, a
triangle or a four-cornered polygon - covers the outline best, and which
way that triangle or quadrilateral points.

Works on YOLO segmentation labels: one file per image, one object per line,
"class x1 y1 x2 y2 ..." with coordinates normalised to 0..1.

No ROS code should exist here.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import os
import re

import cv2
import numpy as np

FAMILY = {
    'round': 'round',
    'triangle_up': 'triangle',
    'triangle_down': 'triangle',
    'square': 'quad',
    'diamond': 'quad',
}

_RF_SUFFIX = re.compile(r"\.rf\.[0-9A-Za-z]+$")
_FRAME_SUFFIX = re.compile(r'_\d+_(jpg|jpeg|png)$', re.I)


# ── data types ──────────────────────────────────────────────────────────────

@dataclass
class LabelObject:
    """One labelled object from a label file."""

    source: str              # original image id (augmented copies share it)
    files: list              # label files of this source image
    class_id: int
    points: np.ndarray       # (N, 2) polygon in pixels
    is_box: bool = False     # the label was a plain box, not an outline


@dataclass
class ShapeFit:
    """How well each simple shape covers an outline."""

    shape: str               # best shape: round, triangle_up/down, square, diamond
    iou: dict                # family -> IoU between the outline and that shape
    solidity: float          # polygon area / convex hull area (< 1: dents or self-crossing)
    area_px: float

    @property
    def family(self):
        return FAMILY[self.shape]

    @property
    def best_iou(self):
        return self.iou[self.family]


@dataclass
class Finding:
    """One labelled object with the problems found."""

    obj: LabelObject
    fit: ShapeFit
    class_shape: str         # usual shape of this object's class
    flags: list = field(default_factory=list)
    detail: str = ''
    margin: float = 0.0      # IoU of the found shape minus IoU of the class's shape

    @property
    def strength(self):
        """How sure a shape flag is: strong, medium or weak (borderline outline)."""
        if self.margin > 0.10:
            return 'strong'
        return 'medium' if self.margin > 0.03 else 'weak'


# ── reading ─────────────────────────────────────────────────────────────────

def source_id(label_name):
    """Roboflow label name -> original image id ('a_jpg.rf.123.txt' -> 'a_jpg')."""
    return _RF_SUFFIX.sub('', os.path.splitext(os.path.basename(label_name))[0])


def name_prefix(source):
    """Original image id -> file-name prefix ('STOP_00012_jpg' -> 'STOP')."""
    return _FRAME_SUFFIX.sub('', source)


def read_labels(label_dir, image_size=(1000, 1000)):
    """
    Read all label files of a directory, keeping one copy per source image.

    Augmented copies of an image (same name before '.rf.<hash>') are read
    once, after checking that their labels agree. Returns
    (objects, empty_sources, copy_mismatches).
    image_size scales the normalised coordinates; any size works for the
    shape test, the real size only makes the pixel numbers meaningful.
    """
    w, h = image_size
    groups = defaultdict(list)
    for name in sorted(os.listdir(label_dir)):
        if name.endswith('.txt'):
            groups[source_id(name)].append(name)

    objects, empty, mismatches = [], [], []
    for src, files in groups.items():
        parsed = [_parse(os.path.join(label_dir, f), w, h) for f in files]
        if len({tuple(c for c, _, _ in p) for p in parsed}) > 1:
            mismatches.append(src)
        first = parsed[0]
        if not first:
            empty.append(src)
        for cls, pts, is_box in first:
            objects.append(LabelObject(src, files, cls, pts, is_box))
    return objects, empty, mismatches


def _parse(path, w, h):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            t = line.split()
            if len(t) == 5:                                   # box: cls cx cy bw bh
                cls, cx, cy, bw, bh = int(t[0]), *map(float, t[1:])
                pts = np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                                [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]])
                out.append((cls, pts * [w, h], True))
            elif len(t) >= 7 and len(t) % 2 == 1:             # polygon
                pts = np.array(list(map(float, t[1:]))).reshape(-1, 2)
                out.append((int(t[0]), pts * [w, h], False))
    return out


# ── shape fitting ───────────────────────────────────────────────────────────

def _iou(poly_a, poly_b, raster=256):
    """Compute the IoU of two polygons by drawing both on a small grid."""
    both = np.vstack([poly_a, poly_b])
    lo = both.min(axis=0)
    scale = (raster - 4) / max(float((both.max(axis=0) - lo).max()), 1e-9)

    def draw(p):
        m = np.zeros((raster, raster), np.uint8)
        cv2.fillPoly(m, [np.round((p - lo) * scale + 2).astype(np.int32)], 1)
        return m

    a, b = draw(poly_a), draw(poly_b)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _quad(hull):
    """Four-cornered polygon approximating the hull."""
    perimeter = cv2.arcLength(hull, True)
    lo, hi = 0.0, 0.25 * perimeter
    for _ in range(40):
        approx = cv2.approxPolyDP(hull, (lo + hi) / 2, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype(np.float64)
        if len(approx) > 4:
            lo = (lo + hi) / 2
        else:
            hi = (lo + hi) / 2
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)


def _triangle_points_up(tri):
    """Check whether a triangle has a single corner at the top (image y points down)."""
    y = np.sort(tri[:, 1])
    return (y[1] - y[0]) > (y[2] - y[1])


def _quad_is_diamond(quad):
    """Check whether the corners sit at top/right/bottom/left, not at the diagonals."""
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    return float(np.mean(np.abs(np.sin(2 * ang)))) < 0.5


def fit_shape(points, simpler_margin=0.02):
    """
    Which simple shape covers the outline best.

    Compares the outline's convex hull with the best ellipse, the smallest
    enclosing triangle and a four-cornered approximation, by IoU. A triangle
    is preferred over a four-cornered shape that fits at most simpler_margin
    better. Returns a ShapeFit, or None for a degenerate outline.
    """
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    hull = cv2.convexHull(pts)
    hull_pts = hull.reshape(-1, 2).astype(np.float64)
    hull_area = cv2.contourArea(hull)
    if len(hull_pts) < 3 or hull_area <= 0:
        return None
    solidity = abs(cv2.contourArea(pts)) / hull_area

    iou = {}
    if len(hull_pts) >= 5:
        (cx, cy), (aw, ah), ang = cv2.fitEllipse(hull)
        t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        ell = np.stack([aw / 2 * np.cos(t), ah / 2 * np.sin(t)], axis=1)
        a = math.radians(ang)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        iou['round'] = _iou(hull_pts, ell @ rot.T + [cx, cy])
    else:
        iou['round'] = 0.0
    _, tri = cv2.minEnclosingTriangle(hull)
    tri = tri.reshape(-1, 2).astype(np.float64)
    iou['triangle'] = _iou(hull_pts, tri)
    quad = _quad(hull)
    iou['quad'] = _iou(hull_pts, quad)

    family = max(iou, key=iou.get)
    # Prefer the simpler shape when it fits almost as well: a four-cornered
    # polygon can cover a triangle by putting one corner on an edge.
    if family == 'quad' and iou['triangle'] >= iou['quad'] - simpler_margin:
        family = 'triangle'
    if family == 'round':
        shape = 'round'
    elif family == 'triangle':
        shape = 'triangle_up' if _triangle_points_up(tri) else 'triangle_down'
    else:
        shape = 'diamond' if _quad_is_diamond(quad) else 'square'
    return ShapeFit(shape, iou, float(solidity), float(hull_area))


# ── audit ───────────────────────────────────────────────────────────────────

def class_shapes(fits_by_class, min_share=0.6):
    """
    Usual shape of every class: its majority shape.

    Returns {class_id: (shape, share)}; shape is None when no shape reaches
    min_share of the class (a class without a clear shape is not checked).
    """
    out = {}
    for cls, shapes in fits_by_class.items():
        if not shapes:
            continue
        shape, n = Counter(shapes).most_common(1)[0]
        share = n / len(shapes)
        out[cls] = (shape if share >= min_share else None, share)
    return out


def prefix_classes(objects, min_share=0.8):
    """Usual class of every file-name prefix, when one class dominates it."""
    by_prefix = defaultdict(list)
    for o in objects:
        by_prefix[name_prefix(o.source)].append(o.class_id)
    out = {}
    for prefix, classes in by_prefix.items():
        cls, n = Counter(classes).most_common(1)[0]
        if n / len(classes) >= min_share:
            out[prefix] = cls
    return out


def audit(objects, min_iou=0.80, min_solidity=0.85, min_area_frac=1e-4,
          image_area=1e6, min_margin=0.03):
    """
    Check every labelled object.

    Geometry flags (from the outline alone):
      shape_mismatch       outline is a different family (round / triangle /
                           four-cornered) than its class usually is, and fits
                           that family better by more than min_margin IoU
                           (coarse hand-drawn polygons are often borderline)
      orientation_mismatch same family, but pointing the other way (triangle
                           up vs down, square vs diamond)
      poor_fit             no simple shape covers the outline well
      self_crossing        polygon has dents or crosses itself
      tiny                 outline is very small
      box_label            plain box in a polygon dataset
    Name flag (independent of geometry):
      name_mismatch        class differs from the usual class of the file's
                           name prefix
    Returns (findings, class_shape_table).
    """
    fits = [fit_shape(o.points) for o in objects]
    by_class = defaultdict(list)
    for o, f in zip(objects, fits):
        if f is not None and not o.is_box:
            by_class[o.class_id].append(f.shape)
    table = class_shapes(by_class)
    usual_class = prefix_classes(objects)

    findings = []
    for o, f in zip(objects, fits):
        expected = table.get(o.class_id, (None, 0))[0]
        finding = Finding(o, f, expected)
        if f is None:
            finding.flags.append('poor_fit')
            finding.detail = 'degenerate outline'
            findings.append(finding)
            continue
        if o.is_box:
            finding.flags.append('box_label')
        elif expected is not None:
            if FAMILY[f.shape] != FAMILY[expected]:
                finding.margin = f.best_iou - f.iou[FAMILY[expected]]
                if finding.margin > min_margin:
                    finding.flags.append('shape_mismatch')
            elif f.shape != expected:
                finding.flags.append('orientation_mismatch')
        if not o.is_box and f.best_iou < min_iou:
            finding.flags.append('poor_fit')
        if f.solidity < min_solidity:
            finding.flags.append('self_crossing')
        if f.area_px < min_area_frac * image_area:
            finding.flags.append('tiny')
        prefix_cls = usual_class.get(name_prefix(o.source))
        if prefix_cls is not None and prefix_cls != o.class_id:
            finding.flags.append('name_mismatch')
            finding.detail = f'name suggests class {prefix_cls}'
        findings.append(finding)
    return findings, table


GEOMETRY_FLAGS = ('shape_mismatch', 'orientation_mismatch')


def score_against_names(findings):
    """
    How well the geometry flags find the name mismatches.

    The file-name check is independent of the geometry, so its mismatches
    serve as a reference set of likely wrong classes. Only mismatches the
    geometry could in principle see (the two classes have different usual
    shapes) count as findable.
    Returns a dict with true positives, misses, extra flags, precision and recall.
    """
    geo = {id(f) for f in findings if any(x in f.flags for x in GEOMETRY_FLAGS)}
    ref = {id(f) for f in findings if 'name_mismatch' in f.flags}
    tp = len(geo & ref)
    return {
        'geometry_flags': len(geo),
        'name_mismatches': len(ref),
        'found': tp,
        'missed': len(ref - geo),
        'extra': len(geo - ref),
        'recall': tp / len(ref) if ref else None,
        'precision': tp / len(geo) if geo else None,
    }
