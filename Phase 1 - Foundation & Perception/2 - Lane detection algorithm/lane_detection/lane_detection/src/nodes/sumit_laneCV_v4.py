#!/usr/bin/env python3
"""
Node name  : sumit_laneCV
File name  : sumit_laneCV.py
Author     : Sumit

System     : Ubuntu 22.04 | JetPack 6.2.2 | ROS2 Humble
             DepthAI 3.6.1 API v3 | OpenCV 4.x | NumPy

Lane Detection Strategy — v4 Drivable-Space Segmentation + dash logic:

    occupied  = yellow tape OR large white-island blobs
    drivable  = NOT occupied, restricted to ROI,
                connected component touching the bottom-center of the BEV

    Heading is computed from the drivable mask itself using a row-wise
    centroid line fit. Dash detection is retained as a secondary cross-check
    and extended with the v4 white-blob dash channel, clean dash-point
    extraction, rolling dash accumulator, centerline polynomial fit, and
    lateral-offset estimate.

Resolution / occupancy-grid logic intentionally preserved from the ROS node:
    UNWARPED_SIZE = (500, 541)
    ROI_TOP       = 0
    OCC_ROWS      = 108
    RESOLUTION    = BEV_REAL_WIDTH_M / OCC_COLS

Video Outputs (saved on node shutdown):
    lane_overlay.mp4
    debug_grid.mp4
    occupancy_grid.mp4
    lane_overlay.h265 / debug_grid.h265 raw reference bitstreams
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from typing import Any

import cv2
import depthai as dai
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String


# ═══════════════════════════════════════════════════════════════════════════
#  GEOMETRY — resolution logic preserved from the ROS node
# ═══════════════════════════════════════════════════════════════════════════

UNWARPED_SIZE = (500, 541)  # preserved: was 500x600 in standalone video script
ORIGINAL_SIZE = (1920, 1080)

SRC = np.array([
    (675, 543),
    (1247, 517),
    (1883, 851),
    (43,  882),
], dtype=np.float32)

# Keep the original destination geometry from the node.  The output dsize is
# still UNWARPED_SIZE=(500,541), so the transform is intentionally cropped.
DST = np.array([
    (0, 0),
    (500, 0),
    (500, 600),
    (0, 600),
], dtype=np.float32)

K = np.array([
    [1459.2276611328125, 0.0, 945.8712768554688],
    [0.0, 1458.7711181640625, 516.719482421875],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

D = np.array([
    -2.480555534362793, -6.659245014190674, 0.00015050164074636996,
    0.0006666610715910792, 22.50022315979004, -2.5853612422943115,
    -6.1252665519714355, 21.745553970336914, 0.0, 0.0, 0.0, 0.0,
    -0.0015395419904962182, 0.0015926380874589086,
], dtype=np.float64)

M = cv2.getPerspectiveTransform(SRC, DST)
M_inv = cv2.getPerspectiveTransform(DST, SRC)

# ── ROI (bird's-eye px) — preserved from ROS node ─────────────────────────
ROI_TOP = 100
ROI_LEFT = 0
ROI_RIGHT = 500

# ── Occupancy grid — preserved from ROS node ──────────────────────────────
OCC_COLS = 100
OCC_ROWS = 108
OCC_DISPLAY_W = 400
OCC_DISPLAY_H = 480
OCC_FREE = 255
OCC_UNKNOWN = 128
OCC_OCCUPIED = 0


# ═══════════════════════════════════════════════════════════════════════════
#  DEBUG MOSAIC LAYOUT  (1920 × 1080 canvas)
# ═══════════════════════════════════════════════════════════════════════════

CANVAS_W = 1920
CANVAS_H = 1080
PAD_OUTER = 16
PAD_INNER = 10
HEADER_H = 40

_AVAIL_W = CANVAS_W - 2 * PAD_OUTER
_AVAIL_H = CANVAS_H - 2 * PAD_OUTER - HEADER_H

TOP_N = 4
TOP_TILE_W = (_AVAIL_W - (TOP_N - 1) * PAD_INNER) // TOP_N
TOP_TILE_H = (_AVAIL_H - PAD_INNER) * 11 // 20

BOT_N = 3
BOT_TILE_W = (_AVAIL_W - (BOT_N - 1) * PAD_INNER) // BOT_N
BOT_TILE_H = _AVAIL_H - TOP_TILE_H - PAD_INNER

TOP_ROW_Y = PAD_OUTER + HEADER_H
BOT_ROW_Y = TOP_ROW_Y + TOP_TILE_H + PAD_INNER

TOP_COL_X = [PAD_OUTER + i * (TOP_TILE_W + PAD_INNER) for i in range(TOP_N)]
BOT_COL_X = [PAD_OUTER + i * (BOT_TILE_W + PAD_INNER) for i in range(BOT_N)]

BG_COLOR = (18, 18, 18)
HEADER_COLOR = (30, 30, 30)
BORDER_COLOR = (55, 55, 55)
LABEL_BG = (0, 0, 0)
LABEL_FG = (0, 220, 220)


# ═══════════════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS (METERS) — preserved from ROS node
# ═══════════════════════════════════════════════════════════════════════════

BEV_REAL_WIDTH_M = 0.37   # road width visible in BEV
BEV_REAL_DEPTH_M = 0.40   # visible depth from camera start
BLIND_SPOT_M = 0.57     # distance from robot foot to camera start

# Square cells, width-accurate.
RESOLUTION = BEV_REAL_WIDTH_M / OCC_COLS   # 0.37/100 = 0.0037 m/cell

# Pixel-per-metre used by the dash accumulator.  This keeps the new dash
# accumulator tied to the ROS node's BEV dimensions/calibration instead of the
# standalone video script's rough 500 px/m defaults.
BEV_PX_PER_M_LAT = UNWARPED_SIZE[0] / BEV_REAL_WIDTH_M
BEV_PX_PER_M_FWD = UNWARPED_SIZE[1] / BEV_REAL_DEPTH_M


# ═══════════════════════════════════════════════════════════════════════════
#  SEGMENTATION CONFIG  (v4 core + updated thresholds from video logic)
# ═══════════════════════════════════════════════════════════════════════════

# Yellow tape.  V_MIN is raised to 180 per the newer CV logic to reject
# darker wood/floor pixels whose hue can overlap yellow tape.
Y_HSV_LOW = np.array([15, 60, 180], dtype=np.uint8)
Y_HSV_HIGH = np.array([32, 255, 255], dtype=np.uint8)

WHITE_S_MAX = 40
WHITE_V_MIN = 175

ISLAND_AREA_HUGE = 3000
ISLAND_AREA_BORDER = 1500
ISLAND_BORDER_MARGIN = 3

BOUND_CLOSE_K = (5, 5)
BOUND_OPEN_K = (3, 3)

HEADING_BAND_H = 140
HEADING_ROW_STEP = 2
HEADING_MIN_ROWS = 5
HEADING_MIN_PIX_PER_ROW = 5

HEADING_SMOOTH_N = 8
HEADING_MAX_JUMP_DEG = 25.0
HEADING_RECOVERY_THR = 6

STATUS_MIN_DRIVABLE_PCT = 5.0


# ═══════════════════════════════════════════════════════════════════════════
#  CENTER DASH CONFIG — edge channel + new blob channel
# ═══════════════════════════════════════════════════════════════════════════

# -- edge channel --
D_CLAHE_CLIP = 1.5
D_CLAHE_GRID = (8, 8)
D_ADAPT_BLOCK = 199
D_ADAPT_C = -45
D_CANNY_LOW = 215
D_CANNY_HIGH = 170

D_MIN_AREA = 120
D_MAX_AREA = 2000
D_MIN_ASPECT = 1.4
D_MIN_SOLIDITY = 0.7

DASH_CLAHE = cv2.createCLAHE(clipLimit=D_CLAHE_CLIP,
                             tileGridSize=D_CLAHE_GRID)

# -- blob channel --
DASH_BLOB_AREA_MIN = 40
DASH_BLOB_AREA_MAX = 6000
DASH_BLOB_MIN_ASPECT = 1.8

# -- clean dash point extraction --
DASH_PT_MAX_ROW_WIDTH = 18
DASH_PT_MIN_THIN_ROWS = 5
DASH_PT_MIN_THIN_COV = 0.60
DASH_PT_MIN_LONG = 15
DASH_PT_MAX_HEADING_DIFF = 25
DASH_PT_BOUNDARY_PROX = 25
DASH_PT_ROI_X_MIN = 20
DASH_PT_ROI_X_MAX = 480


# ═══════════════════════════════════════════════════════════════════════════
#  DASH ACCUMULATOR + CENTERLINE POLYFIT CONFIG
# ═══════════════════════════════════════════════════════════════════════════

DASH_ACCUM_MAX_FRAMES = 20
DASH_ACCUM_BEV_PX_PER_M_FWD = BEV_PX_PER_M_FWD
DASH_ACCUM_BEV_PX_PER_M_LAT = BEV_PX_PER_M_LAT

DASH_POLY_DEGREE = 2
DASH_POLY_MIN_POINTS = 4
DASH_POLY_AGE_WEIGHT = 0.85
DASH_OFFSET_Y_STEP = 10


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════


def build_boundary_mask(bgr_warped: np.ndarray):
    """
    boundary = yellow tape OR large white-island blobs.

    Returns
    -------
    boundary_mask : uint8 mask, 255 = occupied/boundary
    yellow_mask   : uint8 mask, 255 = yellow tape pixels
    white_mask    : uint8 mask, 255 = all near-white pixels
    white_islands : uint8 mask, 255 = white blobs classified as islands
    """
    hsv = cv2.cvtColor(bgr_warped, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, Y_HSV_LOW, Y_HSV_HIGH)
    white_mask = ((hsv[..., 1] < WHITE_S_MAX) &
                  (hsv[..., 2] > WHITE_V_MIN)).astype(np.uint8) * 255

    white_islands = np.zeros_like(white_mask)
    n, labels, stats_, _ = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8)
    h_img, w_img = white_mask.shape
    m = ISLAND_BORDER_MARGIN

    for lbl in range(1, n):
        area = stats_[lbl, cv2.CC_STAT_AREA]
        x = stats_[lbl, cv2.CC_STAT_LEFT]
        y = stats_[lbl, cv2.CC_STAT_TOP]
        w = stats_[lbl, cv2.CC_STAT_WIDTH]
        h = stats_[lbl, cv2.CC_STAT_HEIGHT]
        touches_border = (
            x <= m or y <= ROI_TOP + m or
            x + w >= w_img - m or y + h >= h_img - m
        )
        is_island = (
            area >= ISLAND_AREA_HUGE or
            (area >= ISLAND_AREA_BORDER and touches_border)
        )
        if is_island:
            white_islands[labels == lbl] = 255

    boundary = cv2.bitwise_or(yellow_mask, white_islands)
    boundary = cv2.morphologyEx(
        boundary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, BOUND_OPEN_K))
    boundary = cv2.morphologyEx(
        boundary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, BOUND_CLOSE_K))

    return boundary, yellow_mask, white_mask, white_islands


def extract_dash_mask_white(white_mask: np.ndarray,
                            white_islands: np.ndarray) -> np.ndarray:
    """Blob dash channel: white pixels that are not classified as islands."""
    cand = cv2.bitwise_and(white_mask, cv2.bitwise_not(white_islands))
    cand[:ROI_TOP, :] = 0
    cand[:, :ROI_LEFT] = 0
    cand[:, ROI_RIGHT:] = 0

    out = np.zeros_like(cand)
    n, labels, stats_, _ = cv2.connectedComponentsWithStats(
        cand, connectivity=8)

    for lbl in range(1, n):
        area = stats_[lbl, cv2.CC_STAT_AREA]
        w = stats_[lbl, cv2.CC_STAT_WIDTH]
        h = stats_[lbl, cv2.CC_STAT_HEIGHT]
        if area < DASH_BLOB_AREA_MIN or area > DASH_BLOB_AREA_MAX:
            continue
        aspect = max(h, w) / (min(h, w) + 1e-5)
        if aspect < DASH_BLOB_MIN_ASPECT:
            continue
        out[labels == lbl] = 255

    return out


def extract_dash_points(dash_filt: np.ndarray,
                        heading_deg: float,
                        boundary_mask: np.ndarray) -> list[dict[str, Any]]:
    """Extract clean dash centroid points for accumulated centerline fitting."""
    k = DASH_PT_BOUNDARY_PROX * 2 + 1
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    boundary_dilated = cv2.dilate(boundary_mask, kern)

    n, labels, stats_, _ = cv2.connectedComponentsWithStats(
        dash_filt, connectivity=8)

    pts: list[dict[str, Any]] = []
    for lbl in range(1, n):
        area = int(stats_[lbl, cv2.CC_STAT_AREA])
        if area < 20:
            continue

        ys, xs = np.where(labels == lbl)
        if len(xs) < 2:
            continue

        coords = np.column_stack([xs, ys]).astype(np.float64)
        pts_c = coords - coords.mean(axis=0)
        cov = np.cov(pts_c.T)
        if cov.shape != (2, 2):
            continue

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)
        principal = eigvecs[:, order[-1]]
        long_len = 4.0 * np.sqrt(max(float(eigvals[order[-1]]), 1e-6))
        ang_vert = float(np.degrees(np.arctan2(principal[0], principal[1])))

        diff = abs(ang_vert - heading_deg)
        diff = float(min(diff, abs(diff - 180.0), abs(diff + 180.0)))

        if diff > DASH_PT_MAX_HEADING_DIFF or long_len < DASH_PT_MIN_LONG:
            continue

        thin_xs: list[float] = []
        thin_ys: list[float] = []
        y_min, y_max = int(ys.min()), int(ys.max())
        total_row_span = y_max - y_min + 1

        for y in range(y_min, y_max + 1):
            row_xs = xs[ys == y]
            if len(row_xs) == 0:
                continue
            if row_xs.max() - row_xs.min() + 1 <= DASH_PT_MAX_ROW_WIDTH:
                thin_xs.append(float(row_xs.mean()))
                thin_ys.append(float(y))

        if len(thin_ys) < DASH_PT_MIN_THIN_ROWS:
            continue

        thin_coverage = len(thin_ys) / max(total_row_span, 1)
        if thin_coverage < DASH_PT_MIN_THIN_COV:
            continue

        cx = float(np.mean(thin_xs))
        cy = float(np.mean(thin_ys))

        if cx < DASH_PT_ROI_X_MIN or cx > DASH_PT_ROI_X_MAX:
            continue

        comp_mask = (labels == lbl).astype(np.uint8) * 255
        bfrac = (float(np.count_nonzero(
            cv2.bitwise_and(comp_mask, boundary_dilated))) / max(area, 1))
        if bfrac > 0.3:
            continue

        pts.append({
            'cx': cx,
            'cy': cy,
            'ang_vert': ang_vert,
            'heading_diff': diff,
            'long_len': long_len,
            'n_thin_rows': len(thin_ys),
            'thin_coverage': thin_coverage,
            'thin_xs': thin_xs,
            'thin_ys': thin_ys,
        })

    pts.sort(key=lambda p: -p['cy'])  # nearest first
    return pts


def build_drivable_mask(boundary_mask: np.ndarray):
    """drivable = NOT boundary, restricted to ROI, seed/largest component."""
    roi = np.zeros_like(boundary_mask)
    roi[ROI_TOP:, ROI_LEFT:ROI_RIGHT] = 255

    candidate = cv2.bitwise_and(cv2.bitwise_not(boundary_mask), roi)
    n, labels, stats_, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8)

    seed_x = UNWARPED_SIZE[0] // 2
    seed_y = UNWARPED_SIZE[1] - 1
    seed_label = int(labels[seed_y, seed_x])

    drivable = np.zeros_like(candidate)
    if seed_label != 0:
        drivable[labels == seed_label] = 255
    elif n > 1:
        areas = stats_[1:, cv2.CC_STAT_AREA]
        seed_label = 1 + int(np.argmax(areas))
        drivable[labels == seed_label] = 255

    n_components = max(0, n - 1)
    return drivable, seed_label, n_components


# ═══════════════════════════════════════════════════════════════════════════
#  HEADING ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════


def estimate_heading(drivable: np.ndarray):
    """Primary heading from row-wise centroid line fit of drivable mask."""
    rows: list[int] = []
    centroids: list[float] = []
    y_bottom = UNWARPED_SIZE[1] - 1
    y_top = max(ROI_TOP, UNWARPED_SIZE[1] - HEADING_BAND_H)

    for y in range(y_bottom, y_top - 1, -HEADING_ROW_STEP):
        xs = np.where(drivable[y, :] > 0)[0]
        if len(xs) >= HEADING_MIN_PIX_PER_ROW:
            rows.append(y)
            centroids.append(float(xs.mean()))

    if len(rows) < HEADING_MIN_ROWS:
        return 0.0, np.array(rows), np.array(centroids), False

    rows_arr = np.array(rows, dtype=np.float64)
    cent_arr = np.array(centroids, dtype=np.float64)

    y_max = rows_arr.max()
    y_min = rows_arr.min()
    span = max(y_max - y_min, 1.0)
    weights = 0.2 + 0.8 * (rows_arr - y_min) / span

    a, _b = np.polyfit(rows_arr, cent_arr, 1, w=weights)
    angle_deg = float(np.degrees(np.arctan(a)))
    return angle_deg, rows_arr, cent_arr, True


def estimate_dash_heading(dash_mask: np.ndarray):
    """Secondary heading cross-check from dash pixels within the ROI."""
    rows: list[int] = []
    centroids: list[float] = []
    y_bottom = UNWARPED_SIZE[1] - 1
    y_top = max(ROI_TOP, UNWARPED_SIZE[1] - HEADING_BAND_H)

    n_pts = int(np.count_nonzero(dash_mask[ROI_TOP:, :]))
    if n_pts < 30:
        return None, n_pts

    for y in range(y_bottom, y_top - 1, -HEADING_ROW_STEP):
        xs = np.where(dash_mask[y, :] > 0)[0]
        if len(xs) >= 1:
            rows.append(y)
            centroids.append(float(xs.mean()))

    if len(rows) < HEADING_MIN_ROWS:
        return None, n_pts

    rows_arr = np.array(rows, dtype=np.float64)
    cent_arr = np.array(centroids, dtype=np.float64)
    a, _b = np.polyfit(rows_arr, cent_arr, 1)
    return float(np.degrees(np.arctan(a))), n_pts


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — CENTER DASHES, EDGE CHANNEL
# ═══════════════════════════════════════════════════════════════════════════


def dash_preprocess_edges(gray: np.ndarray):
    """Legacy edge channel preprocessing: CLAHE -> adaptive thresh -> Canny."""
    enhanced = DASH_CLAHE.apply(gray)
    tape_mask = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        D_ADAPT_BLOCK, D_ADAPT_C)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_OPEN, open_k)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    canny = cv2.Canny(blurred, D_CANNY_LOW, D_CANNY_HIGH)
    edges_gated = cv2.bitwise_and(canny, tape_mask)
    return enhanced, edges_gated, tape_mask


def extract_dash_mask_edges(roi_mask: np.ndarray) -> np.ndarray:
    """Legacy edge channel contour filter."""
    closed = cv2.morphologyEx(
        roi_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    dilated = cv2.dilate(
        closed,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=2)

    out = np.zeros_like(dilated)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (D_MIN_AREA <= area <= D_MAX_AREA):
            continue
        _x, _y, w, h = cv2.boundingRect(cnt)
        if (h / (w + 1e-5)) < D_MIN_ASPECT:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0 or (area / hull_area) < D_MIN_SOLIDITY:
            continue
        cv2.drawContours(out, [cnt], -1, 255, -1)

    return out


# ═══════════════════════════════════════════════════════════════════════════
#  OCCUPANCY GRID
# ═══════════════════════════════════════════════════════════════════════════


def build_occupancy_grid(drivable: np.ndarray,
                         boundary: np.ndarray) -> np.ndarray:
    """Build OCC_ROWS x OCC_COLS grid from pixel segmentation."""
    grid = np.full((OCC_ROWS, OCC_COLS), OCC_UNKNOWN, dtype=np.uint8)

    px_per_col = UNWARPED_SIZE[0] / OCC_COLS
    px_per_row = UNWARPED_SIZE[1] / OCC_ROWS

    for row_idx in range(OCC_ROWS):
        y0 = int(row_idx * px_per_row)
        y1 = int((row_idx + 1) * px_per_row)
        if (row_idx + 0.5) * px_per_row < ROI_TOP:
            continue

        for col_idx in range(OCC_COLS):
            x0 = int(col_idx * px_per_col)
            x1 = int((col_idx + 1) * px_per_col)

            cell_drivable = drivable[y0:y1, x0:x1]
            cell_boundary = boundary[y0:y1, x0:x1]
            n_total = cell_drivable.size
            if n_total == 0:
                continue

            n_free = int(np.count_nonzero(cell_drivable))
            n_occ = int(np.count_nonzero(cell_boundary))

            if n_occ / n_total > 0.15:
                grid[row_idx, col_idx] = OCC_OCCUPIED
            elif n_free / n_total > 0.5:
                grid[row_idx, col_idx] = OCC_FREE
            else:
                grid[row_idx, col_idx] = OCC_UNKNOWN

    return grid


# ═══════════════════════════════════════════════════════════════════════════
#  DEBUG MOSAIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _make_tile(img: np.ndarray,
               tw: int,
               th: int,
               label: str,
               metrics: dict | None = None) -> np.ndarray:
    tile = cv2.resize(_to_bgr(img), (tw, th), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), BORDER_COLOR, 1)
    cv2.rectangle(tile, (0, 0), (tw, 24), LABEL_BG, -1)
    cv2.putText(tile, label, (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, LABEL_FG, 1,
                cv2.LINE_AA)
    if metrics is not None:
        dot = ((0, 210, 0) if metrics['status'].startswith('OK')
               else (0, 60, 220))
        cv2.circle(tile, (tw - 14, 12), 6, dot, -1)
    return tile


def _paste(canvas: np.ndarray,
           tile: np.ndarray,
           x: int,
           y: int) -> None:
    th, tw = tile.shape[:2]
    canvas[y:y + th, x:x + tw] = tile


def _first_roi_occ_row() -> int:
    px_per_row = UNWARPED_SIZE[1] / OCC_ROWS
    for r in range(OCC_ROWS):
        if (r + 0.5) * px_per_row >= ROI_TOP:
            return r
    return OCC_ROWS


def render_occ_roi_only(grid: np.ndarray,
                        metrics: dict,
                        tile_w: int,
                        tile_h: int) -> np.ndarray:
    px_per_row = UNWARPED_SIZE[1] / OCC_ROWS
    first_roi_row = _first_roi_occ_row()
    roi_grid = grid[first_roi_row:, :]

    if roi_grid.size == 0:
        roi_grid = np.full((1, OCC_COLS), OCC_UNKNOWN, dtype=np.uint8)

    colour = np.full((*roi_grid.shape, 3), (25, 25, 25), dtype=np.uint8)
    colour[roi_grid == OCC_FREE] = (255, 255, 255)
    colour[roi_grid == OCC_UNKNOWN] = (40, 40, 40)
    colour[roi_grid == OCC_OCCUPIED] = (0, 0, 0)

    legend_h = 56
    grid_h = max(1, tile_h - legend_h)
    n_rows, n_cols = roi_grid.shape

    disp = cv2.resize(colour, (tile_w, grid_h), interpolation=cv2.INTER_NEAREST)

    cell_w = tile_w / max(n_cols, 1)
    cell_h = grid_h / max(n_rows, 1)
    gl = (60, 60, 60)
    for c in range(0, n_cols + 1, 10):
        x = int(c * cell_w)
        cv2.line(disp, (x, 0), (x, grid_h), gl, 1)
    for r in range(0, n_rows + 1, 5):
        y = int(r * cell_h)
        cv2.line(disp, (0, y), (tile_w, y), gl, 1)

    cv2.drawMarker(disp, (tile_w // 2, grid_h - 12),
                   (0, 255, 128), cv2.MARKER_TRIANGLE_UP, 18, 2)

    leg = np.full((legend_h, tile_w, 3), (22, 22, 22), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def swatch(x: int, y: int, col: tuple[int, int, int], txt: str) -> None:
        cv2.rectangle(leg, (x, y), (x + 14, y + 12), col, -1)
        cv2.putText(leg, txt, (x + 18, y + 11), font, 0.36,
                    (200, 200, 200), 1)

    swatch(10, 6, (255, 255, 255), 'Driveable')
    swatch(10, 24, (0, 0, 0), 'Boundary')
    swatch(10, 42, (40, 40, 40), 'Unknown')

    sc = ((0, 210, 0) if metrics['status'].startswith('OK')
          else (0, 90, 255))
    cv2.putText(leg,
                f"{metrics['status']}  "
                f"Heading:{metrics['heading_deg']:+.1f}deg  "
                f"src:{metrics['heading_src']}",
                (140, 14), font, 0.38, sc, 1)
    cv2.putText(leg,
                f"Drivable:{metrics['drivable_pct']:5.1f}%  "
                f"D_pts:{metrics['pts_dash']:3d}  "
                f"ROI rows {first_roi_row}->{OCC_ROWS}",
                (140, 34), font, 0.36, (170, 170, 170), 1)
    cv2.putText(leg,
                f"YellowPts:{metrics['pts_yellow']:5d}  "
                f"WhitePts:{metrics['pts_white']:5d}",
                (140, 52), font, 0.36, (140, 140, 140), 1)

    return np.vstack([disp, leg])


def build_debug_mosaic_1080p(undist: np.ndarray,
                             warped: np.ndarray,
                             boundary_filt: np.ndarray,
                             dash_filt: np.ndarray,
                             seg_vis: np.ndarray,
                             occ_grid: np.ndarray,
                             overlay_frame: np.ndarray,
                             metrics: dict) -> np.ndarray:
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0),
                  (CANVAS_W, PAD_OUTER + HEADER_H - PAD_INNER // 2),
                  HEADER_COLOR, -1)
    dot = ((0, 210, 0) if metrics['status'].startswith('OK')
           else (0, 60, 220))
    cv2.circle(canvas,
               (PAD_OUTER + 14, PAD_OUTER + HEADER_H // 2),
               9, dot, -1)
    cv2.putText(
        canvas,
        (f"sumit_laneCV v4 (drivable-space)  |  "
         f"Status: {metrics['status']}  |  "
         f"Frame: {metrics.get('frame', 0):6d}  |  "
         f"Drivable: {metrics['drivable_pct']:5.1f}%  |  "
         f"Heading: {metrics['heading_deg']:+5.1f}deg "
         f"({metrics['heading_src']})  |  "
         f"D:{metrics['d_mode']} raw={metrics['pts_dash']}px "
         f"pts={metrics['n_dash_pts']} "
         f"poly={'OK' if metrics['poly_valid'] else 'NO'}"),
        (PAD_OUTER + 32, PAD_OUTER + HEADER_H - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1,
        cv2.LINE_AA)

    tw, th = TOP_TILE_W, TOP_TILE_H
    bw, bh = BOT_TILE_W, BOT_TILE_H

    _paste(canvas,
           _make_tile(undist, tw, th, "1: Undistorted", metrics),
           TOP_COL_X[0], TOP_ROW_Y)

    warped_roi = warped.copy()
    cv2.rectangle(warped_roi,
                  (ROI_LEFT, ROI_TOP),
                  (ROI_RIGHT - 1, UNWARPED_SIZE[1] - 1),
                  (0, 220, 220), 2)
    cv2.putText(warped_roi, 'ROI',
                (ROI_LEFT + 4, min(ROI_TOP + 20, UNWARPED_SIZE[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)
    _paste(canvas,
           _make_tile(warped_roi, tw, th, "2: Bird's-Eye + ROI"),
           TOP_COL_X[1], TOP_ROW_Y)

    _paste(canvas,
           _make_tile(boundary_filt, tw, th,
                      "3: Boundary Mask (Yellow+White)"),
           TOP_COL_X[2], TOP_ROW_Y)

    df = np.zeros((*dash_filt.shape, 3), dtype=np.uint8)
    df[dash_filt > 0] = (200, 200, 200)
    _paste(canvas,
           _make_tile(df, tw, th, "4: Dash Lines Filtered"),
           TOP_COL_X[3], TOP_ROW_Y)

    _paste(canvas,
           _make_tile(seg_vis, bw, bh,
                      "5: Drivable Segmentation + Heading"),
           BOT_COL_X[0], BOT_ROW_Y)

    occ_tile = render_occ_roi_only(occ_grid, metrics, bw, bh)
    _paste(canvas,
           _make_tile(occ_tile, bw, bh, "6: Occupancy Grid (ROI only)"),
           BOT_COL_X[1], BOT_ROW_Y)

    _paste(canvas,
           _make_tile(overlay_frame, bw, bh, "7: Drivable Overlay", metrics),
           BOT_COL_X[2], BOT_ROW_Y)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _fmt_num(value: Any, fmt: str, none: str = 'n/a') -> str:
    if value is None:
        return none
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return none


def draw_metrics_panel(img: np.ndarray, m: dict) -> np.ndarray:
    panel = img.copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (620, 410), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)

    c_ok = (0, 255, 0)
    c_warn = (0, 165, 255)
    c_bad = (0, 0, 255)
    c_info = (255, 255, 255)
    c_bnd = (0, 220, 220)
    c_dash = (200, 200, 100)

    def put(t: str, y: int, c: tuple[int, int, int] = c_info) -> None:
        cv2.putText(panel, t, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, c, 1)

    sc = (c_ok if m['status'].startswith('OK') else
          c_warn if m['status'].startswith('PARTIAL') else c_bad)

    put(f"Status       : {m['status']}", 26, sc)
    put(f"Frame        : {m.get('frame', 0):6d}", 50, c_info)

    put("-- DRIVABLE SEGMENTATION ----------------", 78, c_bnd)
    put(f"  drivable area : {m['drivable_pct']:5.1f}%   "
        f"components : {m['n_components']:2d}",
        100,
        c_ok if m['drivable_pct'] >= STATUS_MIN_DRIVABLE_PCT else c_bad)
    put(f"  yellow pts : {m['pts_yellow']:5d}   "
        f"white pts : {m['pts_white']:5d}", 122, c_bnd)

    put("-- HEADING ------------------------------", 150, c_info)
    put(f"  angle : {m['heading_deg']:+6.1f} deg   "
        f"source : {m['heading_src']}   "
        f"rows : {m['heading_rows']:3d}",
        172,
        c_ok if m['heading_src'] != 'NONE' else c_bad)
    put(f"  raw_angle : {m['heading_raw_deg']:+6.1f} deg   "
        f"jump : {m['heading_jump']:5.1f} (max {HEADING_MAX_JUMP_DEG})  "
        f"recovery : {'YES' if m['heading_recovery'] else 'no'}",
        194,
        c_ok if m['heading_jump'] < HEADING_MAX_JUMP_DEG else c_warn)

    dash_ang = m['dash_heading_deg']
    dash_ang_str = 'n/a' if dash_ang is None else f"{dash_ang:+.1f}deg"
    put("-- DASH (secondary heading cross-check) --", 222, c_dash)
    put(f"  mode:{m['d_mode']}  raw:{m['pts_dash']:4d}px  "
        f"blob:{m['pts_dash_blob']:3d}  edge:{m['pts_dash_edge']:3d}  "
        f"clean_pts:{m['n_dash_pts']:2d}  dash_angle:{dash_ang_str}",
        244, c_dash)

    put("-- CONFIG --------------------------------", 272, c_info)
    put(f"  Y_HSV=[{Y_HSV_LOW[0]}-{Y_HSV_HIGH[0]}]  "
        f"WHITE S<{WHITE_S_MAX} V>{WHITE_V_MIN}",
        294, c_info)
    put(f"  DASH_BLOB=[{DASH_BLOB_AREA_MIN},{DASH_BLOB_AREA_MAX}] "
        f"asp>{DASH_BLOB_MIN_ASPECT}  "
        f"HEADING_BAND={HEADING_BAND_H}px smooth={HEADING_SMOOTH_N}",
        316, c_dash)

    put("-- CENTERLINE POLY + LATERAL OFFSET -----", 344, (0, 255, 255))
    pv = bool(m['poly_valid'])
    poly_col = c_ok if pv else c_bad
    put(f"  accum_n={m['accum_n']:3d}  poly={'VALID' if pv else 'NO_FIT':7s}  "
        f"pts_used={m['poly_pts_used']:3d}  pose={m['pose_mode']}",
        366, poly_col)

    le_px = m['lateral_err_px']
    le_m = m['lateral_err_m']
    ml = m['mean_left_px']
    mr = m['mean_right_px']
    if pv and le_px is not None and ml is not None and mr is not None:
        side = 'L_of_C' if le_px > 0 else 'R_of_C'
        put(f"  lat_err={le_px:+6.1f}px ({le_m:+.4f}m) [{side}]  "
            f"L={ml:.1f}px  R={mr:.1f}px",
            388, c_ok)
    else:
        put(f"  lateral offset: n/a (need >={DASH_POLY_MIN_POINTS} accum pts)",
            388, c_warn)

    return panel


def build_seg_vis(boundary: np.ndarray,
                  drivable: np.ndarray,
                  rows: np.ndarray,
                  centroids: np.ndarray,
                  heading_deg: float,
                  heading_valid: bool,
                  dash_points: list[dict[str, Any]] | None = None,
                  centerline_coeffs: np.ndarray | None = None,
                  lateral_offsets: dict | None = None) -> np.ndarray:
    vis = np.full((UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), (35, 35, 35),
                  dtype=np.uint8)
    vis[drivable > 0] = (0, 170, 0)
    vis[boundary > 0] = (0, 0, 200)

    for r, c in zip(rows, centroids):
        cv2.circle(vis, (int(c), int(r)), 2, (255, 255, 0), -1)

    if heading_valid and len(rows) >= 2:
        y0 = float(rows.max())
        y1 = float(rows.min())
        x0 = float(centroids[np.argmax(rows)])
        slope = np.tan(np.radians(heading_deg))
        x1 = x0 + slope * (y1 - y0)
        cv2.arrowedLine(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                        (255, 255, 255), 3, tipLength=0.15)

    if dash_points:
        for dp in dash_points:
            for tx, ty in zip(dp['thin_xs'], dp['thin_ys']):
                cv2.circle(vis, (int(tx), int(ty)), 1, (255, 0, 255), -1)
            cx_i, cy_i = int(dp['cx']), int(dp['cy'])
            cv2.drawMarker(vis, (cx_i, cy_i), (255, 80, 255),
                           cv2.MARKER_CROSS, 14, 2)

    if centerline_coeffs is not None:
        prev_pt = None
        for y in range(UNWARPED_SIZE[1] - 1, ROI_TOP - 1, -3):
            px = int(np.clip(np.polyval(centerline_coeffs, y),
                             0, UNWARPED_SIZE[0] - 1))
            pt = (px, y)
            if prev_pt is not None:
                cv2.line(vis, prev_pt, pt, (0, 255, 255), 2)
            prev_pt = pt

    if lateral_offsets and lateral_offsets.get('rows'):
        for y, px in zip(lateral_offsets['rows'], lateral_offsets['poly_x']):
            cv2.circle(vis, (int(px), int(y)), 3, (0, 140, 255), -1)

    cv2.putText(vis, 'Grn=driv  Red=bnd  Yel=hdg  Mag=dash  Cyn=poly  Org=lat',
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (220, 220, 220), 1)
    return vis


def build_boundary_vis(boundary_mask: np.ndarray,
                       yellow_mask: np.ndarray,
                       white_mask: np.ndarray) -> np.ndarray:
    vis = np.zeros((*boundary_mask.shape, 3), dtype=np.uint8)
    vis[white_mask > 0] = (120, 120, 120)
    vis[yellow_mask > 0] = (0, 215, 255)

    boundary_pixels = boundary_mask > 0
    vis[boundary_pixels] = np.where(
        (vis[boundary_pixels] == 0).all(axis=-1, keepdims=True),
        (0, 0, 200), vis[boundary_pixels])

    highlight = np.zeros_like(vis)
    highlight[boundary_pixels] = (0, 0, 200)
    vis = cv2.addWeighted(vis, 0.5, highlight, 0.5, 0)
    return vis


def render_occupancy_grid(grid: np.ndarray, metrics: dict) -> np.ndarray:
    colour = np.zeros((OCC_ROWS, OCC_COLS, 3), dtype=np.uint8)
    colour[grid == OCC_FREE] = (255, 255, 255)
    colour[grid == OCC_UNKNOWN] = (30, 30, 30)
    colour[grid == OCC_OCCUPIED] = (0, 0, 0)

    disp = cv2.resize(colour, (OCC_DISPLAY_W, OCC_DISPLAY_H),
                      interpolation=cv2.INTER_NEAREST)

    cell_w = OCC_DISPLAY_W / OCC_COLS
    cell_h = OCC_DISPLAY_H / OCC_ROWS
    gc = (60, 60, 60)
    for c in range(OCC_COLS + 1):
        cv2.line(disp, (int(c * cell_w), 0),
                 (int(c * cell_w), OCC_DISPLAY_H), gc, 1)
    for r in range(OCC_ROWS + 1):
        cv2.line(disp, (0, int(r * cell_h)),
                 (OCC_DISPLAY_W, int(r * cell_h)), gc, 1)

    legend = np.zeros((140, OCC_DISPLAY_W, 3), dtype=np.uint8)

    def swatch(x: int, y: int, col: tuple[int, int, int], lbl: str) -> None:
        cv2.rectangle(legend, (x, y), (x + 18, y + 14), col, -1)
        if col == (255, 255, 255):
            cv2.rectangle(legend, (x, y), (x + 18, y + 14),
                          (100, 100, 100), 1)
        cv2.putText(legend, lbl, (x + 22, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

    swatch(10, 6, (255, 255, 255), 'Free / driveable')
    swatch(10, 24, (0, 0, 0), 'Boundary / occupied')
    swatch(10, 42, (30, 30, 30), 'Unknown / outside ROI')

    sc = ((0, 220, 0) if metrics['status'].startswith('OK')
          else (0, 100, 255))
    cv2.putText(legend, f"Status : {metrics['status']}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sc, 1)
    cv2.putText(legend,
                f"Drivable: {metrics['drivable_pct']:.1f}%   "
                f"Heading: {metrics['heading_deg']:+.1f}deg "
                f"({metrics['heading_src']})",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 180, 180), 1)
    cv2.putText(legend,
                f"Yellow pts: {metrics['pts_yellow']:5d}   "
                f"White pts: {metrics['pts_white']:5d}   "
                f"Dash pts: {metrics['pts_dash']:3d}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 180, 180), 1)
    cv2.putText(legend,
                f"Grid {OCC_COLS}x{OCC_ROWS}  "
                f"({UNWARPED_SIZE[0]/OCC_COLS:.1f}x"
                f"{UNWARPED_SIZE[1]/OCC_ROWS:.1f} px/cell)",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (140, 140, 140), 1)

    title = np.zeros((30, OCC_DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(title, 'OCCUPANCY GRID  |  v4 Drivable-Space Pipeline',
                (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 255, 255), 1)

    return np.vstack([title, disp, legend])


def open_mp4_writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer


# ═══════════════════════════════════════════════════════════════════════════
#  DASH ACCUMULATOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def pose_delta(pose_prev: tuple[float, float, float],
               pose_curr: tuple[float, float, float]) -> tuple[float, float, float]:
    """Motion delta in previous body frame: forward, left, yaw."""
    x0, y0, t0 = pose_prev
    x1, y1, t1 = pose_curr

    dx_world = x1 - x0
    dy_world = y1 - y0
    ct, st = np.cos(t0), np.sin(t0)

    dx_fwd = ct * dx_world + st * dy_world
    dy_lat = -st * dx_world + ct * dy_world
    dtheta = t1 - t0
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    return float(dx_fwd), float(dy_lat), float(dtheta)


def compensate_points(points: list[dict[str, Any]],
                      dx_fwd: float,
                      dy_lat: float,
                      dtheta: float) -> list[dict[str, Any]]:
    """Shift accumulated dash points into the current BEV frame."""
    if not points:
        return []

    dpx_x = -dy_lat * DASH_ACCUM_BEV_PX_PER_M_LAT
    dpx_y = -dx_fwd * DASH_ACCUM_BEV_PX_PER_M_FWD

    pivot_x = UNWARPED_SIZE[0] / 2.0
    pivot_y = float(UNWARPED_SIZE[1])

    cos_dt = np.cos(dtheta)
    sin_dt = -np.sin(dtheta)  # image y flipped vs world y

    out: list[dict[str, Any]] = []
    for p in points:
        tx = p['cx'] + dpx_x
        ty = p['cy'] + dpx_y

        rx = tx - pivot_x
        ry = ty - pivot_y
        nx = cos_dt * rx - sin_dt * ry + pivot_x
        ny = sin_dt * rx + cos_dt * ry + pivot_y

        if not (ROI_LEFT <= nx < ROI_RIGHT and
                ROI_TOP <= ny < UNWARPED_SIZE[1]):
            continue

        new_p = dict(p)
        new_p['cx'] = float(nx)
        new_p['cy'] = float(ny)
        out.append(new_p)

    return out


def fit_dash_centerline(accum_points: list[dict[str, Any]],
                        frame_idx: int) -> tuple[np.ndarray | None, int, bool]:
    """Fit polynomial x=f(y) through accumulated dash centroids."""
    if len(accum_points) < DASH_POLY_MIN_POINTS:
        return None, 0, False

    ys = np.array([p['cy'] for p in accum_points], dtype=np.float64)
    xs = np.array([p['cx'] for p in accum_points], dtype=np.float64)
    ages = np.array([frame_idx - p['frame_idx'] for p in accum_points],
                    dtype=np.float64)
    weights = DASH_POLY_AGE_WEIGHT ** ages

    try:
        import warnings
        try:
            from numpy.exceptions import RankWarning as NpRankWarning
        except Exception:
            NpRankWarning = getattr(np, 'RankWarning', Warning)

        with warnings.catch_warnings():
            warnings.simplefilter('error', NpRankWarning)
            try:
                coeffs = np.polyfit(ys, xs, DASH_POLY_DEGREE, w=weights)
            except Exception:
                # Nearly collinear / poorly conditioned: try a linear fit and
                # embed it in the configured polynomial coefficient vector.
                try:
                    coeffs_lin = np.polyfit(ys, xs, 1, w=weights)
                except Exception:
                    return None, 0, False
                coeffs = np.zeros(DASH_POLY_DEGREE + 1, dtype=np.float64)
                coeffs[-2:] = coeffs_lin

        return coeffs, len(accum_points), True

    except (np.linalg.LinAlgError, ValueError):
        return None, 0, False


def compute_lateral_offsets(coeffs: np.ndarray,
                            drivable: np.ndarray) -> dict[str, Any]:
    """Measure fitted dash-centerline offset against drivable boundaries."""
    result: dict[str, Any] = {
        'rows': [],
        'poly_x': [],
        'left_dist': [],
        'right_dist': [],
        'mean_left': None,
        'mean_right': None,
        'lateral_err': None,
    }

    y_vals = range(UNWARPED_SIZE[1] - 1, ROI_TOP - 1, -DASH_OFFSET_Y_STEP)
    for y in y_vals:
        poly_x = float(np.polyval(coeffs, y))
        driv_xs = np.where(drivable[y, :] > 0)[0]
        if len(driv_xs) < 5:
            continue

        left_x = float(driv_xs.min())
        right_x = float(driv_xs.max())
        if not (left_x < poly_x < right_x):
            continue

        result['rows'].append(int(y))
        result['poly_x'].append(poly_x)
        result['left_dist'].append(poly_x - left_x)
        result['right_dist'].append(right_x - poly_x)

    if result['left_dist']:
        ml = float(np.mean(result['left_dist']))
        mr = float(np.mean(result['right_dist']))
        result['mean_left'] = ml
        result['mean_right'] = mr
        result['lateral_err'] = (mr - ml) / 2.0  # + = car left of centre

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  ROS 2 NODE
# ═══════════════════════════════════════════════════════════════════════════


class SumitLaneCV(Node):
    def __init__(self):
        super().__init__('sumit_laneCV')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('camera_fps', 30)
        self.declare_parameter('show_windows', False)
        self.declare_parameter('left_to_center_offset', 230)   # kept for compat
        self.declare_parameter('center_to_right_offset', 230)  # kept for compat
        self.declare_parameter('record_video', True)
        self.declare_parameter('video_out_dir', '~')
        self.declare_parameter('use_odom', True)
        self.declare_parameter('odom_topic', '/odometry/filtered')

        self.cam_fps = int(self.get_parameter('camera_fps').value)
        self.show_win = bool(self.get_parameter('show_windows').value)
        self._record = bool(self.get_parameter('record_video').value)
        self._video_dir = os.path.expanduser(
            str(self.get_parameter('video_out_dir').value))
        self._use_odom = bool(self.get_parameter('use_odom').value)
        self._odom_topic = str(self.get_parameter('odom_topic').value)

        # ── QoS ───────────────────────────────────────────────────────────
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_result = self.create_publisher(
            Image, '/lane/result_image', qos_reliable)
        self.pub_debug = self.create_publisher(
            Image, '/lane/debug_grid', qos_reliable)
        self.pub_occ_img = self.create_publisher(
            Image, '/lane/occupancy_image', qos_reliable)
        self.pub_occ = self.create_publisher(
            OccupancyGrid, '/lane/occupancy_grid', qos_reliable)
        self.pub_coeffs = self.create_publisher(
            Float32MultiArray, '/lane/coefficients', qos_reliable)
        self.pub_status = self.create_publisher(
            String, '/lane/status', qos_reliable)

        # ── Optional odometry subscriber for dash accumulator compensation ─
        self._latest_pose: tuple[float, float, float] | None = None
        self._last_pose: tuple[float, float, float] | None = None
        self._sub_odom = None
        if self._use_odom:
            self._sub_odom = self.create_subscription(
                Odometry, self._odom_topic, self._odom_cb, qos_reliable)

        # ── CV bridge ─────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Heading smoothing/validation state ────────────────────────────
        self._heading_history = deque(maxlen=HEADING_SMOOTH_N)
        self._heading_rejects = 0
        self._heading_recovery = False

        # ── Dash accumulator state ────────────────────────────────────────
        self._dash_accum: list[dict[str, Any]] = []
        self._centerline_coeffs: np.ndarray | None = None
        self._lateral_offsets: dict[str, Any] = {}

        # ── Metrics dict ──────────────────────────────────────────────────
        self._metrics: dict[str, Any] = {
            'status': 'INIT',
            'frame': 0,
            'drivable_pct': 0.0,
            'n_components': 0,
            'pts_yellow': 0,
            'pts_white': 0,
            'pts_dash': 0,
            'heading_deg': 0.0,
            'heading_raw_deg': 0.0,
            'heading_src': 'NONE',
            'heading_rows': 0,
            'heading_jump': 0.0,
            'heading_recovery': False,
            'dash_heading_deg': None,
            'd_mode': 'INIT',
            'pts_dash_blob': 0,
            'pts_dash_edge': 0,
            'dash_points': [],
            'n_dash_pts': 0,
            'accum_n': 0,
            'poly_valid': False,
            'poly_pts_used': 0,
            'lateral_err_px': None,
            'lateral_err_m': None,
            'mean_left_px': None,
            'mean_right_px': None,
            'pose_mode': 'NONE',
        }
        self._frame_idx = 0

        # ── MP4 writers (lazy — opened on first frame) ────────────────────
        self._w_overlay: cv2.VideoWriter | None = None
        self._w_debug: cv2.VideoWriter | None = None
        self._w_occ: cv2.VideoWriter | None = None
        self._writers_ready = False

        # ── H.265 raw bitstream file handles ─────────────────────────────
        self._h265_overlay_file = None
        self._h265_debug_file = None

        # ── DepthAI v3 pipeline ───────────────────────────────────────────
        self._pipeline = None
        self._video_queue = None
        self._enc_overlay_queue = None
        self._enc_debug_queue = None
        self._init_depthai()

        # ── Timer ─────────────────────────────────────────────────────────
        self._timer = self.create_timer(0.01, self._timer_cb)

        self.get_logger().info(
            f'sumit_laneCV v4 ready  |  '
            f'DepthAI {dai.__version__}  |  '
            f'OpenCV {cv2.__version__}  |  '
            f'record={self._record}  dir={self._video_dir}  |  '
            f'odom={self._use_odom} topic={self._odom_topic}  |  '
            f'BEV={UNWARPED_SIZE[0]}x{UNWARPED_SIZE[1]} '
            f'OCC={OCC_COLS}x{OCC_ROWS} res={RESOLUTION:.4f}m'
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Odometry callback
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _yaw_from_quaternion(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._latest_pose = (float(p.x), float(p.y), self._yaw_from_quaternion(q))

    # ──────────────────────────────────────────────────────────────────────
    #  DepthAI v3 pipeline
    # ──────────────────────────────────────────────────────────────────────

    def _init_depthai(self) -> None:
        self._pipeline = dai.Pipeline()

        cam = self._pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_A,
            sensorResolution=(ORIGINAL_SIZE[0], ORIGINAL_SIZE[1]),
            sensorFps=float(self.cam_fps)
        )

        video_out = cam.requestOutput(
            size=(ORIGINAL_SIZE[0], ORIGINAL_SIZE[1]),
            type=dai.ImgFrame.Type.BGR888p
        )

        manip = self._pipeline.create(dai.node.ImageManip)
        manip.initialConfig.addRotateDeg(180)
        manip.setMaxOutputFrameSize(ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1] * 3)
        video_out.link(manip.inputImage)

        self._video_queue = manip.out.createOutputQueue(
            maxSize=1, blocking=False)

        nv12_out = cam.requestOutput(
            size=(ORIGINAL_SIZE[0], ORIGINAL_SIZE[1]),
            type=dai.ImgFrame.Type.NV12
        )

        enc_a = self._pipeline.create(dai.node.VideoEncoder).build(
            nv12_out,
            frameRate=float(self.cam_fps),
            profile=dai.VideoEncoderProperties.Profile.H265_MAIN
        )
        self._enc_overlay_queue = enc_a.out.createOutputQueue(
            maxSize=4, blocking=False)

        enc_b = self._pipeline.create(dai.node.VideoEncoder).build(
            nv12_out,
            frameRate=float(self.cam_fps),
            profile=dai.VideoEncoderProperties.Profile.H265_MAIN
        )
        self._enc_debug_queue = enc_b.out.createOutputQueue(
            maxSize=4, blocking=False)

        self._pipeline.start()

        self.get_logger().info(
            f'DepthAI OAK camera started  '
            f'({ORIGINAL_SIZE[0]}x{ORIGINAL_SIZE[1]} '
            f'@{self.cam_fps}fps)  '
            f'| ImageManip: 180° rotation'
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Lazy MP4 writer initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _init_writers(self,
                      overlay_frame: np.ndarray,
                      occ_frame: np.ndarray) -> None:
        if not self._record:
            self._writers_ready = True
            return

        os.makedirs(self._video_dir, exist_ok=True)

        ov_h, ov_w = overlay_frame.shape[:2]
        occ_h, occ_w = occ_frame.shape[:2]
        fps = float(self.cam_fps)

        try:
            self._w_overlay = open_mp4_writer(
                os.path.join(self._video_dir, 'lane_overlay.mp4'),
                ov_w, ov_h, fps)
            self._w_debug = open_mp4_writer(
                os.path.join(self._video_dir, 'debug_grid.mp4'),
                CANVAS_W, CANVAS_H, fps)
            self._w_occ = open_mp4_writer(
                os.path.join(self._video_dir, 'occupancy_grid.mp4'),
                occ_w, occ_h, fps)
        except RuntimeError as e:
            self.get_logger().error(f'MP4 writer init failed: {e}')
            self._record = False
            self._writers_ready = True
            return

        self._h265_overlay_file = open(
            os.path.join(self._video_dir, 'lane_overlay.h265'), 'wb')
        self._h265_debug_file = open(
            os.path.join(self._video_dir, 'debug_grid.h265'), 'wb')

        self._writers_ready = True
        self.get_logger().info(
            f'Recording started -> {self._video_dir}\n'
            f'  lane_overlay.mp4    [{ov_w}x{ov_h}]\n'
            f'  debug_grid.mp4      [{CANVAS_W}x{CANVAS_H}]\n'
            f'  occupancy_grid.mp4  [{occ_w}x{occ_h}]\n'
            f'  lane_overlay.h265   (raw H.265 camera stream)\n'
            f'  debug_grid.h265     (raw H.265 camera stream)'
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Write frames
    # ──────────────────────────────────────────────────────────────────────

    def _write_frames(self,
                      overlay_frame: np.ndarray,
                      debug_frame: np.ndarray,
                      occ_frame: np.ndarray) -> None:
        if not self._record:
            return
        if self._w_overlay is not None:
            self._w_overlay.write(overlay_frame)
        if self._w_debug is not None:
            self._w_debug.write(debug_frame)
        if self._w_occ is not None:
            self._w_occ.write(occ_frame)

    def _write_h265(self) -> None:
        if not self._record:
            return
        if self._h265_overlay_file is not None and self._enc_overlay_queue is not None:
            while self._enc_overlay_queue.has():
                pkt = self._enc_overlay_queue.get()
                if pkt is not None:
                    pkt.getData().tofile(self._h265_overlay_file)

        if self._h265_debug_file is not None and self._enc_debug_queue is not None:
            while self._enc_debug_queue.has():
                pkt = self._enc_debug_queue.get()
                if pkt is not None:
                    pkt.getData().tofile(self._h265_debug_file)

    # ──────────────────────────────────────────────────────────────────────
    #  Timer callback
    # ──────────────────────────────────────────────────────────────────────

    def _timer_cb(self) -> None:
        if self._pipeline is None or not self._pipeline.isRunning():
            return

        in_rgb = self._video_queue.get() if self._video_queue.has() else None
        if in_rgb is None:
            return

        frame = in_rgb.getCvFrame()  # already 180°-rotated by ImageManip
        result, dbg, occ_display, occ_grid = self._detect_lane(frame)

        if not self._writers_ready:
            self._init_writers(result, occ_display)

        self._write_frames(result, dbg, occ_display)
        self._write_h265()

        stamp = self.get_clock().now().to_msg()

        self.pub_result.publish(
            self.bridge.cv2_to_imgmsg(result, encoding='bgr8'))
        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
        self.pub_occ_img.publish(
            self.bridge.cv2_to_imgmsg(occ_display, encoding='bgr8'))

        self.pub_occ.publish(self._to_ros_occ_grid(occ_grid, stamp))

        # Compatibility publisher: keep six values and ordering from old node.
        m = self._metrics
        dim = MultiArrayDimension(label='lane_coeffs', size=6, stride=6)
        msg_c = Float32MultiArray()
        msg_c.layout.dim = [dim]
        msg_c.data = [
            float(m['heading_deg']),
            float(m['heading_raw_deg']),
            0.0,
            float(m['drivable_pct']),
            float(m['pts_yellow']),
            float(m['pts_dash']),
        ]
        self.pub_coeffs.publish(msg_c)

        s_msg = String()
        s_msg.data = self._status_json(m)
        self.pub_status.publish(s_msg)

        if self.show_win:
            cv2.imshow('sumit_laneCV | Final', cv2.resize(result, (1280, 720)))
            cv2.imshow('sumit_laneCV | Debug', cv2.resize(dbg, (1280, 720)))
            cv2.imshow('sumit_laneCV | Occupancy', occ_display)
            cv2.waitKey(1)

    # ──────────────────────────────────────────────────────────────────────
    #  Status JSON helper
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _float_or_none(v: Any) -> float | None:
        if v is None:
            return None
        return float(v)

    def _status_json(self, m: dict[str, Any]) -> str:
        payload = {
            'status': m['status'],
            'frame': int(m['frame']),
            'drivable_pct': round(float(m['drivable_pct']), 1),
            'n_components': int(m['n_components']),
            'pts_yellow': int(m['pts_yellow']),
            'pts_white': int(m['pts_white']),
            'pts_dash': int(m['pts_dash']),
            'pts_dash_blob': int(m['pts_dash_blob']),
            'pts_dash_edge': int(m['pts_dash_edge']),
            'n_dash_pts': int(m['n_dash_pts']),
            'heading_deg': round(float(m['heading_deg']), 2),
            'heading_raw_deg': round(float(m['heading_raw_deg']), 2),
            'heading_src': m['heading_src'],
            'heading_rows': int(m['heading_rows']),
            'heading_jump': round(float(m['heading_jump']), 2),
            'heading_recovery': bool(m['heading_recovery']),
            'dash_heading_deg': self._float_or_none(m['dash_heading_deg']),
            'd_mode': m['d_mode'],
            'accum_n': int(m['accum_n']),
            'poly_valid': bool(m['poly_valid']),
            'poly_pts_used': int(m['poly_pts_used']),
            'lateral_err_px': self._float_or_none(m['lateral_err_px']),
            'lateral_err_m': self._float_or_none(m['lateral_err_m']),
            'mean_left_px': self._float_or_none(m['mean_left_px']),
            'mean_right_px': self._float_or_none(m['mean_right_px']),
            'pose_mode': m['pose_mode'],
        }
        return json.dumps(payload, separators=(',', ':'))

    # ──────────────────────────────────────────────────────────────────────
    #  OccupancyGrid -> ROS message
    # ──────────────────────────────────────────────────────────────────────

    def _to_ros_occ_grid(self, occ_grid, stamp) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'base_footprint_imu'

        msg.info.resolution = RESOLUTION

        # How many OCC rows fall above ROI_TOP (i.e. are ignored)
        px_per_row    = UNWARPED_SIZE[1] / OCC_ROWS          # 600/100 = 6.0
        ignored_rows  = int(ROI_TOP / px_per_row)            # int(100/6) = 0
        roi_occ_grid  = occ_grid[ignored_rows:, :]           # (83, 100)
        roi_rows      = roi_occ_grid.shape[0]                # 100

        # After rot90+fliplr the grid becomes (OCC_COLS, roi_rows) = (100, 100)
        # ROS width  = number of cells along Y axis  = roi_rows  (depth cells become Y after rotation)
        # ROS height = number of cells along X axis  = OCC_COLS
        # BUT: conventionally width=cols, height=rows BEFORE rotation.
        # Set them to match the rotated output shape:
        rotated_shape_rows = OCC_COLS    # 100  (becomes ROS height)
        rotated_shape_cols = roi_rows    # 100   (becomes ROS width)

        msg.info.width  = rotated_shape_cols   # 100
        msg.info.height = rotated_shape_rows   # 100

        # Physical size of published grid
        total_depth_m = roi_rows  * RESOLUTION   # 83 * 0.0037 = 0.37 m  (< 0.40 m because cells are width-matched)
        total_width_m = OCC_COLS  * RESOLUTION   # 100 * 0.0037 = 0.37 m

        # Origin: lower-left corner of the grid in base_footprint frame
        # X forward = blind spot distance (grid starts where camera vision starts)
        # Y left    = half the road width to the left
        msg.info.origin          = Pose()
        msg.info.origin.position.x = BLIND_SPOT_M           #  0.725 m forward
        msg.info.origin.position.y = -(total_width_m / 2.0) # -0.185 m (centred laterally)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        ros = np.full((roi_rows, OCC_COLS), -1, dtype=np.int8)
        ros[roi_occ_grid == OCC_FREE]     = 0
        ros[roi_occ_grid == OCC_OCCUPIED] = 100

        rotated_ros = np.fliplr(np.rot90(ros, k=1))
        msg.data = rotated_ros.flatten().tolist()
        return msg

    # ──────────────────────────────────────────────────────────────────────
    #  MAIN DETECTION PIPELINE
    # ──────────────────────────────────────────────────────────────────────

    def _compensate_dash_accum_from_pose(self) -> str:
        pose_mode = 'NONE'
        pose = self._latest_pose if self._use_odom else None

        if pose is not None:
            if self._last_pose is not None:
                dx_fwd, dy_lat, dtheta = pose_delta(self._last_pose, pose)
                self._dash_accum = compensate_points(
                    self._dash_accum, dx_fwd, dy_lat, dtheta)
                pose_mode = 'POSE'
            self._last_pose = pose
            if pose_mode == 'NONE':
                pose_mode = 'POSE'  # first frame with pose; no delta yet

        cutoff = self._frame_idx - DASH_ACCUM_MAX_FRAMES
        self._dash_accum = [p for p in self._dash_accum
                            if p['frame_idx'] >= cutoff]
        return pose_mode

    def _detect_lane(self, frame: np.ndarray):
        self._frame_idx += 1

        # Ego-motion compensation for accumulated dash points.
        pose_mode = self._compensate_dash_accum_from_pose()

        # ── Resize / undistort / warp ─────────────────────────────────────
        if (frame.shape[1], frame.shape[0]) != ORIGINAL_SIZE:
            frame = cv2.resize(frame, ORIGINAL_SIZE)
        undist = cv2.undistort(frame, K, D)
        warped = cv2.warpPerspective(
            undist, M, UNWARPED_SIZE, flags=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # ══════════════════════════════════════════════════════════════════
        #  SEGMENTATION
        # ══════════════════════════════════════════════════════════════════
        boundary_mask, yellow_mask, white_mask, white_islands = \
            build_boundary_mask(warped)

        drivable, _seed_label, n_components = build_drivable_mask(boundary_mask)

        roi_area = (UNWARPED_SIZE[1] - ROI_TOP) * (ROI_RIGHT - ROI_LEFT)
        drivable_pct = 100.0 * float(np.count_nonzero(drivable)) / max(roi_area, 1)

        pts_yellow = int(np.count_nonzero(yellow_mask[ROI_TOP:, :]))
        pts_white = int(np.count_nonzero(white_mask[ROI_TOP:, :]))

        # ══════════════════════════════════════════════════════════════════
        #  DASH DETECTION — blob channel + edge channel, unioned
        # ══════════════════════════════════════════════════════════════════
        dash_blob = extract_dash_mask_white(white_mask, white_islands)

        _enhanced, edges_gated, _tape_mask = dash_preprocess_edges(gray)
        dash_roi = edges_gated.copy()
        dash_roi[:ROI_TOP, :] = 0
        dash_roi[:, :ROI_LEFT] = 0
        dash_roi[:, ROI_RIGHT:] = 0
        dash_edge = extract_dash_mask_edges(dash_roi)

        dash_filt = cv2.bitwise_or(dash_blob, dash_edge)
        pts_dash = int(np.count_nonzero(dash_filt))
        pts_dash_blob = int(np.count_nonzero(dash_blob))
        pts_dash_edge = int(np.count_nonzero(dash_edge))
        d_mode = 'DETECT' if pts_dash > 0 else 'LOW_PTS'

        # ══════════════════════════════════════════════════════════════════
        #  HEADING — primary from drivable mask
        # ══════════════════════════════════════════════════════════════════
        raw_angle, h_rows, h_centroids, h_valid = estimate_heading(drivable)

        heading_jump = 0.0
        if h_valid:
            if self._heading_history:
                last = float(np.mean(self._heading_history))
                heading_jump = abs(raw_angle - last)
                if heading_jump <= HEADING_MAX_JUMP_DEG or self._heading_recovery:
                    self._heading_history.append(raw_angle)
                    self._heading_rejects = 0
                    self._heading_recovery = False
                else:
                    self._heading_rejects += 1
                    if self._heading_rejects >= HEADING_RECOVERY_THR:
                        self._heading_recovery = True
                        self._heading_history.clear()
                        self._heading_history.append(raw_angle)
            else:
                self._heading_history.append(raw_angle)
                self._heading_rejects = 0
                self._heading_recovery = False
        else:
            self._heading_rejects += 1
            if self._heading_rejects >= HEADING_RECOVERY_THR:
                self._heading_recovery = True
                self._heading_history.clear()

        if self._heading_history:
            heading_deg = float(np.mean(self._heading_history))
            heading_src = 'DRIVABLE_PCA'
        else:
            heading_deg = 0.0
            heading_src = 'NONE'

        dash_points = extract_dash_points(dash_filt, heading_deg, boundary_mask)
        dash_heading_deg, _dash_n_pts = estimate_dash_heading(dash_filt)

        # ══════════════════════════════════════════════════════════════════
        #  DASH ACCUMULATOR + CENTERLINE POLYFIT
        # ══════════════════════════════════════════════════════════════════
        for dp in dash_points:
            self._dash_accum.append({
                'cx': dp['cx'],
                'cy': dp['cy'],
                'ang_vert': dp['ang_vert'],
                'heading_diff': dp['heading_diff'],
                'frame_idx': self._frame_idx,
            })

        coeffs, poly_pts_used, poly_valid = fit_dash_centerline(
            self._dash_accum, self._frame_idx)
        if poly_valid:
            self._centerline_coeffs = coeffs
            self._lateral_offsets = compute_lateral_offsets(coeffs, drivable)
        else:
            self._centerline_coeffs = None
            self._lateral_offsets = {}

        lat = self._lateral_offsets
        lat_err_px = lat.get('lateral_err', None)
        lat_err_m = (lat_err_px / DASH_ACCUM_BEV_PX_PER_M_LAT
                     if lat_err_px is not None else None)

        # ══════════════════════════════════════════════════════════════════
        #  OCCUPANCY GRID
        # ══════════════════════════════════════════════════════════════════
        occ_grid = build_occupancy_grid(drivable, boundary_mask)

        # ══════════════════════════════════════════════════════════════════
        #  STATUS
        # ══════════════════════════════════════════════════════════════════
        if drivable_pct < STATUS_MIN_DRIVABLE_PCT:
            status = 'NO_DRIVABLE_AREA'
        elif not h_valid and not self._heading_history:
            status = 'PARTIAL_NO_HEADING'
        elif self._heading_recovery:
            status = 'PARTIAL_HEADING_RECOVERY'
        else:
            status = 'OK'

        # ══════════════════════════════════════════════════════════════════
        #  METRICS
        # ══════════════════════════════════════════════════════════════════
        self._metrics.update({
            'status': status,
            'frame': self._frame_idx,
            'drivable_pct': drivable_pct,
            'n_components': n_components,
            'pts_yellow': pts_yellow,
            'pts_white': pts_white,
            'pts_dash': pts_dash,
            'heading_deg': heading_deg,
            'heading_raw_deg': raw_angle,
            'heading_src': heading_src,
            'heading_rows': int(len(h_rows)),
            'heading_jump': heading_jump,
            'heading_recovery': self._heading_recovery,
            'dash_heading_deg': dash_heading_deg,
            'd_mode': d_mode,
            'pts_dash_blob': pts_dash_blob,
            'pts_dash_edge': pts_dash_edge,
            'dash_points': dash_points,
            'n_dash_pts': len(dash_points),
            'accum_n': len(self._dash_accum),
            'poly_valid': poly_valid,
            'poly_pts_used': poly_pts_used,
            'lateral_err_px': lat_err_px,
            'lateral_err_m': lat_err_m,
            'mean_left_px': lat.get('mean_left', None),
            'mean_right_px': lat.get('mean_right', None),
            'pose_mode': pose_mode,
        })

        # ══════════════════════════════════════════════════════════════════
        #  OUTPUTS
        # ══════════════════════════════════════════════════════════════════
        occ_display = render_occupancy_grid(occ_grid, self._metrics)

        seg_vis = build_seg_vis(
            boundary_mask, drivable,
            h_rows, h_centroids,
            heading_deg, h_valid or bool(self._heading_history),
            dash_points=dash_points,
            centerline_coeffs=self._centerline_coeffs,
            lateral_offsets=self._lateral_offsets)

        boundary_vis = build_boundary_vis(boundary_mask, yellow_mask, white_mask)

        # ── drivable overlay -> unwarp back to camera view ────────────────
        bird_ov = np.zeros((UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3),
                           dtype=np.uint8)
        bird_ov[drivable > 0] = (0, 180, 0)
        bird_ov[boundary_mask > 0] = (0, 0, 200)

        if (h_valid or self._heading_history) and len(h_rows) >= 2:
            y0 = float(h_rows.max())
            y1 = float(h_rows.min())
            x0 = float(h_centroids[np.argmax(h_rows)])
            slope = np.tan(np.radians(heading_deg))
            x1 = x0 + slope * (y1 - y0)
            cv2.arrowedLine(bird_ov, (int(x0), int(y0)), (int(x1), int(y1)),
                            (255, 255, 255), 4, tipLength=0.15)

        overlay_inv = cv2.warpPerspective(
            bird_ov, M_inv, ORIGINAL_SIZE, flags=cv2.INTER_LINEAR)
        overlay_frame = cv2.addWeighted(undist, 1.0, overlay_inv, 0.45, 0)
        overlay_frame = draw_metrics_panel(overlay_frame, self._metrics)

        dbg = build_debug_mosaic_1080p(
            undist=undist,
            warped=warped,
            boundary_filt=boundary_vis,
            dash_filt=dash_filt,
            seg_vis=seg_vis,
            occ_grid=occ_grid,
            overlay_frame=overlay_frame,
            metrics=self._metrics)

        return overlay_frame, dbg, occ_display, occ_grid

    # ──────────────────────────────────────────────────────────────────────
    #  Cleanup
    # ──────────────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        for attr, label in (('_w_overlay', 'lane_overlay.mp4'),
                            ('_w_debug', 'debug_grid.mp4'),
                            ('_w_occ', 'occupancy_grid.mp4')):
            writer = getattr(self, attr, None)
            if writer is not None:
                writer.release()
                self.get_logger().info(f'Saved {label}')

        self._write_h265()
        for attr, label in (('_h265_overlay_file', 'lane_overlay.h265'),
                            ('_h265_debug_file', 'debug_grid.h265')):
            f = getattr(self, attr, None)
            if f is not None:
                f.close()
                self.get_logger().info(f'Saved {label}')

        if self.show_win:
            cv2.destroyAllWindows()

        try:
            if self._pipeline is not None and self._pipeline.isRunning():
                self._pipeline.stop()
        except Exception:
            pass

        super().destroy_node()


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SumitLaneCV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()