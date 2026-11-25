#!/usr/bin/env python3

"""
depth_estimator.py.

Distance of a detection, read from the OAK-D depth image.

No ROS code should exist here.
"""

import numpy as np


def bbox_distance(depth_mm, bbox, image_shape, half_size=3):
    """
    Median depth in metres of a small patch at the centre of a box.

    depth_mm    : HxW depth image in millimetres (0 = no depth)
    bbox        : [x1, y1, x2, y2] in pixels of the RGB image
    image_shape : shape of the RGB image the box refers to
    half_size   : the patch is (2h+1)x(2h+1) depth pixels (h=3 -> 7x7)

    The depth image is assumed to cover the same field of view as the RGB
    image (depth aligned to RGB). When its resolution differs, the box centre
    is scaled to depth pixels instead of being used as-is.

    Returns None when the patch holds no valid depth.
    """
    x1, y1, x2, y2 = bbox
    depth_h, depth_w = depth_mm.shape[:2]
    image_h, image_w = image_shape[:2]

    cx = int((x1 + x2) / 2 * depth_w / image_w)
    cy = int((y1 + y2) / 2 * depth_h / image_h)
    cx = min(max(cx, 0), depth_w - 1)
    cy = min(max(cy, 0), depth_h - 1)

    h = half_size
    patch = depth_mm[max(0, cy - h):cy + h + 1, max(0, cx - h):cx + h + 1]

    valid = patch[patch > 0]
    if valid.size == 0:
        return None

    return float(np.median(valid)) / 1000.0
