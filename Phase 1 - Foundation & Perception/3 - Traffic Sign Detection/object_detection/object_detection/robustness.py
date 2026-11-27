#!/usr/bin/env python3

"""
robustness.py.

Image corruptions for testing how well a sign model copes with bad
conditions: darkness, low contrast, glare, motion blur and sensor noise,
each at three strengths (severity 1 = mild, 3 = strong).

The corruptions change only the pixels, never the geometry, so the labels of
the original image stay valid and no relabelling is needed.

No ROS code should exist here.
"""

import cv2
import numpy as np

CORRUPTIONS = ('dark', 'low_contrast', 'glare', 'motion_blur', 'noise')

# strength per severity 1, 2, 3
_LEVELS = {
    'dark': (0.6, 0.4, 0.25),            # brightness factor
    'low_contrast': (0.6, 0.4, 0.25),    # contrast factor around the mean
    'glare': (90, 150, 210),             # added brightness at the centre of the blob
    'motion_blur': (7, 13, 21),          # horizontal kernel length (px)
    'noise': (10, 20, 35),               # Gaussian noise sigma (grey levels)
}


def corrupt(image, name, severity, rng):
    """
    Return a corrupted copy of a BGR uint8 image.

    name     : one of CORRUPTIONS
    severity : 1, 2 or 3
    rng      : numpy Generator; glare position and noise come from it, so a
               fixed seed gives the same corrupted image every time
    """
    if name not in _LEVELS:
        raise ValueError(f'Unknown corruption {name!r}; use one of {CORRUPTIONS}')
    if severity not in (1, 2, 3):
        raise ValueError('severity must be 1, 2 or 3')
    level = _LEVELS[name][severity - 1]
    img = image.astype(np.float32)
    h, w = img.shape[:2]

    if name == 'dark':
        out = img * level
    elif name == 'low_contrast':
        mean = img.mean(axis=(0, 1), keepdims=True)
        out = (img - mean) * level + mean
    elif name == 'glare':
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        r = 0.2 * min(h, w)
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        out = img + level * blob[..., None]
    elif name == 'motion_blur':
        kernel = np.zeros((level, level), np.float32)
        kernel[level // 2, :] = 1.0 / level
        out = cv2.filter2D(img, -1, kernel)
    else:                                                    # noise
        out = img + rng.normal(0, level, img.shape)

    return np.clip(out, 0, 255).astype(np.uint8)
