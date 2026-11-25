#!/usr/bin/env python3

"""
utils.py

Common utility functions used throughout the object_detection package.
"""

import cv2


def rotate_image_180(frame):
    """
    Rotate image by 180 degrees.
    """
    return cv2.rotate(frame, cv2.ROTATE_180)


def bbox_center(bbox):
    """
    Compute center of bounding box.

    bbox = [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox

    cx = int((xmin + xmax) / 2)
    cy = int((ymin + ymax) / 2)

    return cx, cy


def format_confidence(confidence):
    """
    Convert confidence (0.0-1.0) to a percentage string.
    """
    return f"{confidence * 100:.1f} %"


def format_distance(distance):
    """
    Format distance in metres.
    """
    return f"{distance:.2f} m"


def draw_label(
    image,
    text,
    x,
    y,
    color=(0, 255, 0),
):
    """
    Draw a label with black background.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2

    (w, h), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    cv2.rectangle(
        image,
        (x, y - h - 8),
        (x + w + 8, y + baseline),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (x + 4, y - 4),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def latency_summary(latencies_s, arrivals_s):
    """
    Summarise message latencies.

    latencies_s : receive time minus header stamp, per message (seconds)
    arrivals_s  : receive time of each message (seconds)

    Returns a dict with the message count, rate (Hz) and latency mean,
    p50 / p95 / p99 and max in milliseconds, or None for no messages.
    The percentiles show jitter: a node with a good mean can still stall.
    """
    import numpy as np

    if len(latencies_s) == 0:
        return None

    ms = 1000.0 * np.asarray(latencies_s, dtype=float)
    span = max(arrivals_s) - min(arrivals_s) if len(arrivals_s) > 1 else 0.0
    rate = (len(arrivals_s) - 1) / span if span > 0 else 0.0
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])

    return {
        "count": len(ms),
        "rate_hz": rate,
        "mean_ms": float(ms.mean()),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(ms.max()),
    }


def clamp(value, minimum, maximum):
    """
    Clamp value between min and max.
    """

    return max(minimum, min(value, maximum))
