#!/usr/bin/env python3
"""
Node name  : sumit_laneCV
File name  : sumit_laneCV.py
Author     : Sumit
System     : Ubuntu 22.04 | JetPack 6.2.2 | ROS2 Humble
             DepthAI 3.6.1 API v3 | OpenCV 4.13.0 | NumPy 1.23.5

Lane Detection Strategy:
    PRIMARY   : Yellow tape on left edge  (HSV color mask)   — v2 logic
    SECONDARY : Center dashed lines       (v1 adaptive threshold + dash filter)
                When center dashes missing → estimated from yellow + offset
                Output smoother prevents snapping between modes
    RIGHT     : Always offset-based       (no CV tracking)

Video Outputs (saved on node shutdown):
    lane_overlay.mp4                     — polygon overlay on road view
    debug_grid.mp4  (1920×1080 mosaic)   — 7-panel debug mosaic
    occupancy_grid.mp4                   — occupancy grid only
    lane_overlay.h265 / debug_grid.h265  — raw H.265 reference bitstreams
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos  import (QoSProfile,
                         ReliabilityPolicy,
                         HistoryPolicy)

from sensor_msgs.msg   import Image
from std_msgs.msg      import Float32MultiArray, MultiArrayDimension, String
from nav_msgs.msg      import OccupancyGrid
from geometry_msgs.msg import Pose

from cv_bridge import CvBridge

import cv2
import numpy   as np
import depthai as dai
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
#  GEOMETRY — shared
# ═══════════════════════════════════════════════════════════════════════════

UNWARPED_SIZE = (500, 600)
ORIGINAL_SIZE = (1920, 1080)

SRC = np.array(
    [(743, 413), (1237, 419), (1904, 806), (10, 738)],
    dtype=np.float32
)
DST = np.array(
    [(0, 0), (500, 0), (500, 600), (0, 600)],
    dtype=np.float32
)

K = np.array([
    [796.8074340820312, 0.0,               632.9873046875      ],
    [0.0,               796.5176391601562, 431.61431884765625  ],
    [0.0,               0.0,               1.0                 ]
], dtype=np.float64)

D = np.array([
     24.26119041442871,    -44.0109977722168,
     -0.001162048545666039, -4.417484433361096e-06,
     26.147001266479492,    23.662704467773438,
    -42.24923324584961,     24.82693099975586,
      0.0, 0.0, 0.0, 0.0,
     -0.010093020275235176, -0.008346530608832836
], dtype=np.float64)

M     = cv2.getPerspectiveTransform(SRC, DST)
M_inv = cv2.getPerspectiveTransform(DST, SRC)

# ── Lane geometry offsets (bird's-eye px) ─────────────────────────────────
L2C_OFFSET = 230    # yellow left  → center
C2R_OFFSET = 230    # center       → right virtual edge

# ── ROI (bird's-eye px) ───────────────────────────────────────────────────
ROI_TOP   = 280
ROI_LEFT  = 0
ROI_RIGHT = 485

# ── Occupancy grid ────────────────────────────────────────────────────────
OCC_COLS      = 100
OCC_ROWS      = 120
OCC_DISPLAY_W = 400
OCC_DISPLAY_H = 480
OCC_FREE      = 255
OCC_UNKNOWN   = 128
OCC_OCCUPIED  = 0

# ══════════════════════════════════════════════════════════════════════════
#  DEBUG MOSAIC LAYOUT  (1920 × 1080 canvas)
# ══════════════════════════════════════════════════════════════════════════

CANVAS_W     = 1920
CANVAS_H     = 1080
PAD_OUTER    = 16
PAD_INNER    = 10
HEADER_H     = 40

_AVAIL_W     = CANVAS_W - 2 * PAD_OUTER
_AVAIL_H     = CANVAS_H - 2 * PAD_OUTER - HEADER_H

TOP_N        = 4
TOP_TILE_W   = (_AVAIL_W - (TOP_N - 1) * PAD_INNER) // TOP_N
TOP_TILE_H   = (_AVAIL_H - PAD_INNER) * 11 // 20

BOT_N        = 3
BOT_TILE_W   = (_AVAIL_W - (BOT_N - 1) * PAD_INNER) // BOT_N
BOT_TILE_H   = _AVAIL_H - TOP_TILE_H - PAD_INNER

TOP_ROW_Y    = PAD_OUTER + HEADER_H
BOT_ROW_Y    = TOP_ROW_Y + TOP_TILE_H + PAD_INNER

TOP_COL_X    = [PAD_OUTER + i * (TOP_TILE_W + PAD_INNER) for i in range(TOP_N)]
BOT_COL_X    = [PAD_OUTER + i * (BOT_TILE_W + PAD_INNER) for i in range(BOT_N)]

BG_COLOR     = (18,  18,  18)
HEADER_COLOR = (30,  30,  30)
BORDER_COLOR = (55,  55,  55)
LABEL_BG     = (0,   0,   0)
LABEL_FG     = (0,  220, 220)


# ═══════════════════════════════════════════════════════════════════════════
#  YELLOW TAPE CONFIG  (v2 — PRIMARY, left edge)
# ═══════════════════════════════════════════════════════════════════════════

Y_HSV_LOW  = np.array([ 18,  80,  80], dtype=np.uint8)
Y_HSV_HIGH = np.array([ 40, 255, 255], dtype=np.uint8)

Y_OPEN_K   = (5,  5)
Y_CLOSE_K  = (15, 15)
Y_MIN_AREA = 200
Y_MIN_PTS  = 25

Y_SEARCH_WINDOW          = 120
Y_SEARCH_WINDOW_RECOVERY = 400
Y_MAX_LAT_JUMP           = 150
Y_MAX_CURV_JUMP          = 0.15
Y_SMOOTH_N               = 5
Y_RECOVERY_THR           = 8


# ═══════════════════════════════════════════════════════════════════════════
#  CENTER DASH CONFIG  (v1 — SECONDARY, white dashed line)
# ═══════════════════════════════════════════════════════════════════════════

D_CLAHE_CLIP  = 1.5
D_CLAHE_GRID  = (8, 8)
D_ADAPT_BLOCK = 199
D_ADAPT_C     = -45
D_CANNY_LOW   = 255
D_CANNY_HIGH  = 255

D_MIN_AREA     = 120
D_MAX_AREA     = 2000
D_MIN_ASPECT   = 1.4
D_MIN_SOLIDITY = 0.7

D_SEARCH_WINDOW          = 100
D_SEARCH_WINDOW_RECOVERY = 400
D_MAX_LAT_JUMP           = 150
D_MAX_CURV_JUMP          = 0.15
D_SMOOTH_N               = 5
D_RECOVERY_THR           = 8

D_OFFSET_FALLBACK_THR = 4
D_OUTPUT_SMOOTH_N     = 8
D_OUTPUT_MAX_LAT      = 60

_DASH_CLAHE = cv2.createCLAHE(
    clipLimit=D_CLAHE_CLIP, tileGridSize=D_CLAHE_GRID)


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — YELLOW TAPE  (v2 logic)
# ═══════════════════════════════════════════════════════════════════════════

def yellow_extract_mask(bgr_warped: np.ndarray) -> np.ndarray:
    hsv  = cv2.cvtColor(bgr_warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, Y_HSV_LOW, Y_HSV_HIGH)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, Y_OPEN_K))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, Y_CLOSE_K))
    return mask


def yellow_filter_contours(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= Y_MIN_AREA:
            cv2.drawContours(out, [cnt], -1, 255, -1)
    return out


def yellow_fit(filtered: np.ndarray, last_x, recovery: bool):
    hist   = np.sum(filtered, axis=0).astype(np.float32)
    window = Y_SEARCH_WINDOW_RECOVERY if recovery else Y_SEARCH_WINDOW

    if last_x is not None:
        lo = max(0, int(last_x) - window)
        hi = min(UNWARPED_SIZE[0], int(last_x) + window)
        w2 = np.zeros_like(hist)
        w2[lo:hi] = hist[lo:hi]
        if w2.max() >= 5:
            hist = w2

    if hist.max() < 5:
        return None, np.array([]), np.array([]), last_x

    peak   = int(np.argmax(hist))
    lo     = max(0, peak - window)
    hi     = min(UNWARPED_SIZE[0], peak + window)
    locked = np.zeros_like(filtered)
    locked[:, lo:hi] = filtered[:, lo:hi]

    ys, xs = np.where(locked > 0)
    if len(ys) < Y_MIN_PTS:
        return None, ys, xs, last_x

    coeffs = np.polyfit(ys, xs, 2)
    bot    = float(np.polyval(coeffs, UNWARPED_SIZE[1] - 1))
    return coeffs, ys, xs, bot


def yellow_is_valid(new_c: np.ndarray, history: deque):
    if len(history) == 0:
        return True, 0.0, 0.0
    last      = np.mean(history, axis=0)
    lat_jump  = abs(np.polyval(new_c, UNWARPED_SIZE[1] - 1)
                    - np.polyval(last, UNWARPED_SIZE[1] - 1))
    curv_jump = abs(new_c[0] - last[0])
    valid     = (lat_jump <= Y_MAX_LAT_JUMP and
                 curv_jump <= Y_MAX_CURV_JUMP)
    return valid, float(lat_jump), float(curv_jump)


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — CENTER DASHES  (v1 logic)
# ═══════════════════════════════════════════════════════════════════════════

def dash_preprocess(gray: np.ndarray):
    enhanced    = _DASH_CLAHE.apply(gray)
    tape_mask   = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        D_ADAPT_BLOCK, D_ADAPT_C)
    open_k      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tape_mask   = cv2.morphologyEx(tape_mask, cv2.MORPH_OPEN, open_k)
    blurred     = cv2.GaussianBlur(enhanced, (5, 5), 0)
    canny       = cv2.Canny(blurred, D_CANNY_LOW, D_CANNY_HIGH)
    edges_gated = cv2.bitwise_and(canny, tape_mask)
    return enhanced, edges_gated, tape_mask


def dash_filter_contours(roi_mask: np.ndarray) -> np.ndarray:
    closed  = cv2.morphologyEx(
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
        _, _, w, h = cv2.boundingRect(cnt)
        if (h / (w + 1e-5)) < D_MIN_ASPECT:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0 or (area / hull_area) < D_MIN_SOLIDITY:
            continue
        cv2.drawContours(out, [cnt], -1, 255, -1)
    return out


def dash_find_center_x(filtered: np.ndarray, last_cx, recovery: bool):
    hist   = np.sum(filtered, axis=0).astype(np.float32)
    window = D_SEARCH_WINDOW_RECOVERY if recovery else D_SEARCH_WINDOW

    if last_cx is not None:
        lo = max(0, last_cx - window)
        hi = min(UNWARPED_SIZE[0], last_cx + window)
        w2 = np.zeros_like(hist)
        w2[lo:hi] = hist[lo:hi]
        if w2.max() >= 5:
            hist = w2

    if hist.max() < 5:
        return last_cx
    return int(np.argmax(hist))


def dash_fit(filtered: np.ndarray, center_x, recovery: bool):
    if center_x is None:
        locked = filtered
    else:
        window = D_SEARCH_WINDOW_RECOVERY if recovery else D_SEARCH_WINDOW
        lo     = max(0, center_x - window)
        hi     = min(UNWARPED_SIZE[0], center_x + window)
        locked = np.zeros_like(filtered)
        locked[:, lo:hi] = filtered[:, lo:hi]

    ys, xs = np.where(locked > 0)
    if len(ys) < 20:
        return None, ys, xs

    coeffs = np.polyfit(ys, xs, 2)
    return coeffs, ys, xs


def dash_is_valid(new_c: np.ndarray, history: deque):
    if len(history) == 0:
        return True, 0.0, 0.0
    last      = np.mean(history, axis=0)
    lat_jump  = abs(np.polyval(new_c, UNWARPED_SIZE[1] - 1)
                    - np.polyval(last, UNWARPED_SIZE[1] - 1))
    curv_jump = abs(new_c[0] - last[0])
    valid     = (lat_jump <= D_MAX_LAT_JUMP and
                 curv_jump <= D_MAX_CURV_JUMP)
    return valid, float(lat_jump), float(curv_jump)


# ═══════════════════════════════════════════════════════════════════════════
#  DEBUG MOSAIC HELPERS  (v3 — 1920×1080 7-panel layout)
# ═══════════════════════════════════════════════════════════════════════════

def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert grayscale to BGR if needed; pass colour through unchanged."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _make_tile(img:     np.ndarray,
               tw:      int,
               th:      int,
               label:   str,
               metrics: dict = None) -> np.ndarray:
    """
    Resize *img* to (tw × th), stamp a label bar, add an optional
    status dot.  Always returns a 3-channel BGR tile.
    """
    tile = cv2.resize(_to_bgr(img), (tw, th),
                      interpolation=cv2.INTER_AREA)
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
           tile:   np.ndarray,
           x:      int,
           y:      int) -> None:
    """Blit *tile* onto *canvas* at pixel offset (x, y)."""
    th, tw = tile.shape[:2]
    canvas[y:y + th, x:x + tw] = tile


def render_occ_roi_only(grid:    np.ndarray,
                        metrics: dict,
                        tile_w:  int,
                        tile_h:  int) -> np.ndarray:
    """
    Render the occupancy grid clipped to bird's-eye ROI rows,
    with a legend strip.  Returns a (tile_h × tile_w) BGR image.
    """
    px_per_row = UNWARPED_SIZE[1] / OCC_ROWS

    first_roi_row = 0
    for r in range(OCC_ROWS):
        if (r + 0.5) * px_per_row >= ROI_TOP:
            first_roi_row = r
            break

    roi_grid = grid[first_roi_row:, :]

    colour = np.full((*roi_grid.shape, 3), (25, 25, 25), dtype=np.uint8)
    colour[roi_grid == OCC_FREE]     = (230, 230, 230)
    colour[roi_grid == OCC_UNKNOWN]  = (40,  40,  40)
    colour[roi_grid == OCC_OCCUPIED] = (0,   80,  200)

    legend_h = 56
    grid_h   = tile_h - legend_h
    n_rows, n_cols = roi_grid.shape

    disp = cv2.resize(colour, (tile_w, grid_h),
                      interpolation=cv2.INTER_NEAREST)

    cell_w = tile_w / n_cols
    cell_h = grid_h / n_rows
    gl     = (60, 60, 60)
    for c in range(0, n_cols + 1, 10):
        x = int(c * cell_w)
        cv2.line(disp, (x, 0), (x, grid_h), gl, 1)
    for r in range(0, n_rows + 1, 5):
        y = int(r * cell_h)
        cv2.line(disp, (0, y), (tile_w, y), gl, 1)

    cv2.drawMarker(disp, (tile_w // 2, grid_h - 12),
                   (0, 255, 128), cv2.MARKER_TRIANGLE_UP, 18, 2)

    leg = np.full((legend_h, tile_w, 3), (22, 22, 22), dtype=np.uint8)
    F   = cv2.FONT_HERSHEY_SIMPLEX

    def swatch(x, y, col, txt):
        cv2.rectangle(leg, (x, y), (x + 14, y + 12), col, -1)
        cv2.putText(leg, txt, (x + 18, y + 11), F, 0.36,
                    (200, 200, 200), 1)

    swatch(10,  6, (230, 230, 230), 'Driveable')
    swatch(10, 24, (0,   80,  200), 'Boundary')
    swatch(10, 42, (40,  40,   40), 'Unknown')

    sc = ((0, 210, 0) if metrics['status'].startswith('OK')
          else (0, 90, 255))
    cv2.putText(leg,
                f"{metrics['status']}  "
                f"Y:{metrics['y_mode']}  D:{metrics['d_mode']}",
                (140, 14), F, 0.38, sc, 1)
    cv2.putText(leg,
                f"Y_pts:{metrics['pts_yellow']:3d}  "
                f"D_pts:{metrics['pts_dash']:3d}  "
                f"ROI rows {first_roi_row}→{OCC_ROWS}",
                (140, 34), F, 0.36, (170, 170, 170), 1)
    cv2.putText(leg,
                f"L2C={metrics.get('l2c', L2C_OFFSET)}px  "
                f"C2R={metrics.get('c2r', C2R_OFFSET)}px",
                (140, 52), F, 0.36, (140, 140, 140), 1)

    return np.vstack([disp, leg])


def build_debug_mosaic_1080p(undist:        np.ndarray,
                              warped:        np.ndarray,
                              yellow_filt:   np.ndarray,
                              dash_filt:     np.ndarray,
                              fit_vis:       np.ndarray,
                              occ_grid:      np.ndarray,
                              overlay_frame: np.ndarray,
                              metrics:       dict) -> np.ndarray:
    """
    Compose a 1920×1080 debug mosaic from 7 panels:

      Top row  (4 tiles): undistorted | bird's-eye+ROI |
                          yellow-filtered | dash-filtered
      Bottom row (3 tiles): poly-fit vis | occ grid | lane overlay
    """
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

    # ── header bar ────────────────────────────────────────────────────────
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
        (f"sumit_laneCV v3  |  "
         f"Status: {metrics['status']}  |  "
         f"Frame: {metrics.get('frame', 0):6d}  |  "
         f"Y:{metrics['y_mode']} ({metrics['pts_yellow']}pts)  |  "
         f"D:{metrics['d_mode']} ({metrics['pts_dash']}pts)  |  "
         f"Y_rec:{'Y' if metrics['y_recovery'] else 'n'}  "
         f"D_rec:{'Y' if metrics['d_recovery'] else 'n'}"),
        (PAD_OUTER + 32, PAD_OUTER + HEADER_H - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1,
        cv2.LINE_AA)

    tw, th = TOP_TILE_W, TOP_TILE_H
    bw, bh = BOT_TILE_W, BOT_TILE_H

    # ── top row ───────────────────────────────────────────────────────────

    # panel 1 — undistorted
    _paste(canvas,
           _make_tile(undist, tw, th, "1: Undistorted", metrics),
           TOP_COL_X[0], TOP_ROW_Y)

    # panel 2 — bird's-eye with ROI rectangle
    warped_roi = warped.copy()
    cv2.rectangle(warped_roi,
                  (ROI_LEFT, ROI_TOP),
                  (ROI_RIGHT, UNWARPED_SIZE[1] - 1),
                  (0, 220, 220), 2)
    cv2.putText(warped_roi, 'ROI',
                (ROI_LEFT + 4, ROI_TOP + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)
    _paste(canvas,
           _make_tile(warped_roi, tw, th, "2: Bird's-Eye + ROI"),
           TOP_COL_X[1], TOP_ROW_Y)

    # panel 3 — yellow mask false-colour
    yf = np.zeros((*yellow_filt.shape, 3), dtype=np.uint8)
    yf[yellow_filt > 0] = (0, 215, 255)
    _paste(canvas,
           _make_tile(yf, tw, th, "3: Yellow Tape Filtered"),
           TOP_COL_X[2], TOP_ROW_Y)

    # panel 4 — dash mask false-colour
    df = np.zeros((*dash_filt.shape, 3), dtype=np.uint8)
    df[dash_filt > 0] = (200, 200, 200)
    _paste(canvas,
           _make_tile(df, tw, th, "4: Dash Lines Filtered"),
           TOP_COL_X[3], TOP_ROW_Y)

    # ── bottom row ────────────────────────────────────────────────────────

    # panel 5 — polynomial fit vis
    _paste(canvas,
           _make_tile(fit_vis, bw, bh,
                      "5: Poly Fit  Yellow=left  Cyan=center  Red=right"),
           BOT_COL_X[0], BOT_ROW_Y)

    # panel 6 — occupancy grid (ROI rows only)
    occ_tile = render_occ_roi_only(occ_grid, metrics, bw, bh)
    _paste(canvas,
           _make_tile(occ_tile, bw, bh, "6: Occupancy Grid (ROI only)"),
           BOT_COL_X[1], BOT_ROW_Y)

    # panel 7 — final lane overlay
    _paste(canvas,
           _make_tile(overlay_frame, bw, bh, "7: Lane Overlay", metrics),
           BOT_COL_X[2], BOT_ROW_Y)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════
#  OCCUPANCY GRID
# ═══════════════════════════════════════════════════════════════════════════

def build_occupancy_grid(left_coeffs,
                         right_x_arr: np.ndarray,
                         ploty:       np.ndarray) -> np.ndarray:
    grid       = np.full((OCC_ROWS, OCC_COLS), OCC_UNKNOWN, dtype=np.uint8)
    px_per_col = UNWARPED_SIZE[0] / OCC_COLS
    px_per_row = UNWARPED_SIZE[1] / OCC_ROWS

    if left_coeffs is None:
        return grid

    for row_idx in range(OCC_ROWS):
        bird_y = (row_idx + 0.5) * px_per_row
        if bird_y < ROI_TOP:
            continue

        lx  = float(np.polyval(left_coeffs, bird_y))
        idx = int(np.clip(np.searchsorted(ploty, bird_y),
                          0, len(right_x_arr) - 1))
        rx  = float(right_x_arr[idx])

        for col_idx in range(OCC_COLS):
            bird_x = (col_idx + 0.5) * px_per_col
            grid[row_idx, col_idx] = (
                OCC_FREE if lx <= bird_x <= rx else OCC_OCCUPIED)

        if 0.0 <= lx < UNWARPED_SIZE[0]:
            lx_c = int(lx / px_per_col)
            for dc in range(-1, 2):
                c = lx_c + dc
                if 0 <= c < OCC_COLS:
                    grid[row_idx, c] = OCC_OCCUPIED

        if 0.0 <= rx < UNWARPED_SIZE[0]:
            rx_c = int(rx / px_per_col)
            for dc in range(-1, 2):
                c = rx_c + dc
                if 0 <= c < OCC_COLS:
                    grid[row_idx, c] = OCC_OCCUPIED

    return grid


def render_occupancy_grid(grid: np.ndarray, metrics: dict) -> np.ndarray:
    """Standalone full occupancy grid image for occupancy_grid.mp4."""
    colour = np.zeros((OCC_ROWS, OCC_COLS, 3), dtype=np.uint8)
    colour[grid == OCC_FREE]     = (255, 255, 255)
    colour[grid == OCC_UNKNOWN]  = (30,  30,  30)
    colour[grid == OCC_OCCUPIED] = (0,   0,   0)

    disp = cv2.resize(colour, (OCC_DISPLAY_W, OCC_DISPLAY_H),
                      interpolation=cv2.INTER_NEAREST)

    cell_w = OCC_DISPLAY_W / OCC_COLS
    cell_h = OCC_DISPLAY_H / OCC_ROWS
    gc     = (50, 50, 50)
    for c in range(OCC_COLS + 1):
        cv2.line(disp, (int(c * cell_w), 0),
                 (int(c * cell_w), OCC_DISPLAY_H), gc, 1)
    for r in range(OCC_ROWS + 1):
        cv2.line(disp, (0, int(r * cell_h)),
                 (OCC_DISPLAY_W, int(r * cell_h)), gc, 1)

    legend = np.zeros((130, OCC_DISPLAY_W, 3), dtype=np.uint8)

    def swatch(x, y, col, lbl):
        cv2.rectangle(legend, (x, y), (x + 18, y + 14), col, -1)
        if col == (255, 255, 255):
            cv2.rectangle(legend, (x, y), (x + 18, y + 14),
                          (100, 100, 100), 1)
        cv2.putText(legend, lbl, (x + 22, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

    swatch(10,  6, (255, 255, 255), 'Free / driveable')
    swatch(10, 24, (0,   0,   0),   'Boundary / occupied')
    swatch(10, 42, (30,  30,  30),  'Unknown / outside ROI')

    sc = ((0, 220, 0) if metrics['status'].startswith('OK')
          else (0, 100, 255))
    cv2.putText(legend, f"Status : {metrics['status']}",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sc, 1)
    cv2.putText(legend,
                f"Yellow: {metrics['y_mode']}  Dash: {metrics['d_mode']}",
                (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(legend,
                f"Yellow pts: {metrics['pts_yellow']:3d}   "
                f"Dash pts: {metrics['pts_dash']:3d}",
                (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(legend,
                f"Grid {OCC_COLS}x{OCC_ROWS}  "
                f"({UNWARPED_SIZE[0]/OCC_COLS:.1f}x"
                f"{UNWARPED_SIZE[1]/OCC_ROWS:.1f} px/cell)",
                (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

    title = np.zeros((28, OCC_DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(title, 'OCCUPANCY GRID  |  v3 Yellow+Dash Pipeline',
                (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    return np.vstack([title, disp, legend])


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def draw_metrics_panel(img: np.ndarray, m: dict) -> np.ndarray:
    panel   = img.copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (560, 390), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)

    C_OK   = (0, 255, 0)
    C_WARN = (0, 165, 255)
    C_BAD  = (0, 0, 255)
    C_INFO = (255, 255, 255)
    C_YEL  = (0, 220, 220)
    C_DASH = (200, 200, 100)

    def put(t, y, c=C_INFO):
        cv2.putText(panel, t, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, c, 1)

    sc = (C_OK   if m['status'].startswith('OK')  else
          C_WARN if m['status'].startswith('REJ') else C_BAD)

    put(f"Status       : {m['status']}",  26, sc)

    put("── YELLOW (primary left edge) ──────────────",  54, C_YEL)
    put(f"  mode:{m['y_mode']}  pts:{m['pts_yellow']:3d}  "
        f"rec:{'YES' if m['y_recovery'] else 'no'}",
        76,
        C_OK if m['pts_yellow'] >= Y_MIN_PTS else C_BAD)
    put(f"  A={m['y_A']:+.6f}  B={m['y_B']:+.4f}  "
        f"bot={m['y_bot']:.1f}px", 98, C_YEL)
    put(f"  lat={m['y_lat']:.2f}px(max {Y_MAX_LAT_JUMP})  "
        f"curv={m['y_curv']:.5f}(max {Y_MAX_CURV_JUMP})",
        120,
        C_OK if m['y_lat'] < Y_MAX_LAT_JUMP else C_BAD)

    put("── DASH  (secondary center) ────────────────", 148, C_DASH)
    put(f"  mode:{m['d_mode']}  pts:{m['pts_dash']:3d}  "
        f"rec:{'YES' if m['d_recovery'] else 'no'}",
        170,
        C_OK if m['d_mode'] == 'DETECT' else C_WARN)
    put(f"  A={m['d_A']:+.6f}  B={m['d_B']:+.4f}", 192, C_DASH)
    put(f"  lat={m['d_lat']:.2f}px(max {D_MAX_LAT_JUMP})  "
        f"curv={m['d_curv']:.5f}(max {D_MAX_CURV_JUMP})",
        214,
        C_OK if m['d_lat'] < D_MAX_LAT_JUMP else C_BAD)
    put(f"  out_smoother: N={D_OUTPUT_SMOOTH_N}  "
        f"max_lat={D_OUTPUT_MAX_LAT}px  "
        f"fallback_thr={D_OFFSET_FALLBACK_THR}",
        236, C_DASH)

    put("── RIGHT (offset from center) ──────────────", 264, C_INFO)
    put(f"  L2C={m.get('l2c', L2C_OFFSET)}px   "
        f"C2R={m.get('c2r', C2R_OFFSET)}px", 286, C_INFO)

    put(f"Y: HSV[{Y_HSV_LOW[0]}-{Y_HSV_HIGH[0]}] "
        f"smooth={Y_SMOOTH_N} rec_thr={Y_RECOVERY_THR}",
        314, C_YEL)
    put(f"D: adaptB={D_ADAPT_BLOCK} C={D_ADAPT_C} "
        f"smooth={D_SMOOTH_N} rec_thr={D_RECOVERY_THR}",
        336, C_DASH)
    put(f"D: area=[{D_MIN_AREA},{D_MAX_AREA}] "
        f"asp={D_MIN_ASPECT} sol={D_MIN_SOLIDITY}",
        358, C_DASH)
    put(f"Frame: {m.get('frame', 0):6d}", 380, C_INFO)

    return panel


def build_fit_vis(ys_y, xs_y, ys_d, xs_d,
                  left_x, center_x, right_x,
                  ploty, l_valid, c_valid, r_valid) -> np.ndarray:
    vis = np.zeros((UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)

    if len(ys_y) > 0:
        ys_c = np.clip(ys_y, 0, UNWARPED_SIZE[1] - 1)
        xs_c = np.clip(xs_y, 0, UNWARPED_SIZE[0] - 1)
        vis[ys_c, xs_c] = (0, 215, 255)

    if len(ys_d) > 0:
        ys_c = np.clip(ys_d, 0, UNWARPED_SIZE[1] - 1)
        xs_c = np.clip(xs_d, 0, UNWARPED_SIZE[0] - 1)
        vis[ys_c, xs_c] = (200, 200, 200)

    for i in range(len(ploty)):
        cy = int(ploty[i])
        if not (0 <= cy < UNWARPED_SIZE[1]):
            continue
        if l_valid[i]:
            lxi = int(left_x[i])
            if 0 <= lxi < UNWARPED_SIZE[0]:
                cv2.circle(vis, (lxi, cy), 2, (0, 215, 255), -1)
        if c_valid[i]:
            cxi = int(center_x[i])
            if 0 <= cxi < UNWARPED_SIZE[0]:
                cv2.circle(vis, (cxi, cy), 2, (255, 255, 0), -1)
        if r_valid[i]:
            rxi = int(right_x[i])
            if 0 <= rxi < UNWARPED_SIZE[0]:
                cv2.circle(vis, (rxi, cy), 2, (0, 0, 255), -1)

    cv2.putText(vis, 'Yellow=left  Cyan=center  Red=right',
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return vis


# ═══════════════════════════════════════════════════════════════════════════
#  MP4 WRITER HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _open_mp4_writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer


# ═══════════════════════════════════════════════════════════════════════════
#  ROS 2 NODE
# ═══════════════════════════════════════════════════════════════════════════

class SumitLaneCV(Node):
    """
    ROS 2 Humble lane-detection node — v3 dual pipeline.

    Detection hierarchy:
        1. Yellow tape   → LEFT edge  (v2 HSV, independent config)
        2. Center dashes → CENTER     (v1 adaptive threshold, independent config)
                           Output smoother prevents DETECT↔OFFSET snapping
           Fallback      → CENTER = LEFT + L2C_OFFSET
        3. Right edge    → RIGHT = CENTER + C2R_OFFSET (always)

    Published topics:
        /lane/result_image      sensor_msgs/Image
        /lane/debug_grid        sensor_msgs/Image   (1920×1080 mosaic)
        /lane/occupancy_image   sensor_msgs/Image
        /lane/occupancy_grid    nav_msgs/OccupancyGrid
        /lane/coefficients      std_msgs/Float32MultiArray
        /lane/status            std_msgs/String  (JSON)

    Saved videos:
        lane_overlay.mp4
        debug_grid.mp4        (always 1920×1080)
        occupancy_grid.mp4
        lane_overlay.h265
        debug_grid.h265
    """

    def __init__(self):
        super().__init__('sumit_laneCV')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('camera_fps',             30)
        self.declare_parameter('show_windows',           False)
        self.declare_parameter('left_to_center_offset',  L2C_OFFSET)
        self.declare_parameter('center_to_right_offset', C2R_OFFSET)
        self.declare_parameter('record_video',           True)
        self.declare_parameter('video_out_dir',          '~')

        self.cam_fps    = int(self.get_parameter('camera_fps').value)
        self.show_win   = bool(self.get_parameter('show_windows').value)
        self.l2c        = int(self.get_parameter('left_to_center_offset').value)
        self.c2r        = int(self.get_parameter('center_to_right_offset').value)
        self._record    = bool(self.get_parameter('record_video').value)
        self._video_dir = os.path.expanduser(
            self.get_parameter('video_out_dir').value)

        # ── QoS ───────────────────────────────────────────────────────────
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_result  = self.create_publisher(
            Image,             '/lane/result_image',    qos_reliable)
        self.pub_debug   = self.create_publisher(
            Image,             '/lane/debug_grid',      qos_reliable)
        self.pub_occ_img = self.create_publisher(
            Image,             '/lane/occupancy_image', qos_reliable)
        self.pub_occ     = self.create_publisher(
            OccupancyGrid,     '/lane/occupancy_grid',  qos_reliable)
        self.pub_coeffs  = self.create_publisher(
            Float32MultiArray, '/lane/coefficients',    qos_reliable)
        self.pub_status  = self.create_publisher(
            String,            '/lane/status',          qos_reliable)

        # ── CV bridge ─────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── Algorithm state — YELLOW (v2) ─────────────────────────────────
        self._y_history  = deque(maxlen=Y_SMOOTH_N)
        self._y_last_x   = None
        self._y_rejects  = 0
        self._y_recovery = False

        # ── Algorithm state — DASH (v1) ───────────────────────────────────
        self._d_history        = deque(maxlen=D_SMOOTH_N)
        self._d_last_cx        = None
        self._d_rejects        = 0
        self._d_recovery       = False
        self._d_consec_rejects = 0

        # ── Output smoother ───────────────────────────────────────────────
        self._center_out_history = deque(maxlen=D_OUTPUT_SMOOTH_N)

        # ── Metrics dict — includes l2c / c2r from the start ─────────────
        self._metrics = {
            'status':     'INIT', 'frame':      0,
            'l2c':        self.l2c,
            'c2r':        self.c2r,
            'y_mode':     'INIT', 'pts_yellow': 0,
            'y_recovery': False,  'y_bot':      0.0,
            'y_A':        0.0,    'y_B':        0.0,
            'y_lat':      0.0,    'y_curv':     0.0,
            'd_mode':     'INIT', 'pts_dash':   0,
            'd_recovery': False,
            'd_A':        0.0,    'd_B':        0.0,
            'd_lat':      0.0,    'd_curv':     0.0,
        }
        self._frame_idx = 0

        # ── MP4 writers (lazy — opened on first frame) ────────────────────
        self._w_overlay     : cv2.VideoWriter | None = None
        self._w_debug       : cv2.VideoWriter | None = None
        self._w_occ         : cv2.VideoWriter | None = None
        self._writers_ready : bool                   = False

        # ── H.265 raw bitstream file handles ─────────────────────────────
        self._h265_overlay_file = None
        self._h265_debug_file   = None

        # ── DepthAI v3 pipeline ───────────────────────────────────────────
        self._pipeline          = None
        self._video_queue       = None
        self._enc_overlay_queue = None
        self._enc_debug_queue   = None
        self._init_depthai()

        # ── Timer ─────────────────────────────────────────────────────────
        self._timer = self.create_timer(0.01, self._timer_cb)

        self.get_logger().info(
            f'sumit_laneCV v3 ready  |  '
            f'DepthAI {dai.__version__}  |  '
            f'OpenCV {cv2.__version__}  |  '
            f'L2C={self.l2c}px  C2R={self.c2r}px  |  '
            f'record={self._record}  dir={self._video_dir}'
        )

    # ──────────────────────────────────────────────────────────────────────
    #  DepthAI v3 pipeline
    # ──────────────────────────────────────────────────────────────────────

    def _init_depthai(self):
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
            f'@ {self.cam_fps}fps)  '
            f'| ImageManip: 180° rotation'
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Lazy MP4 writer initialisation
    # ──────────────────────────────────────────────────────────────────────

    def _init_writers(self,
                      overlay_frame: np.ndarray,
                      occ_frame:     np.ndarray) -> None:
        """
        Open all VideoWriters on the first processed frame.
        The debug writer is always CANVAS_W × CANVAS_H (1920×1080).
        overlay and occ sizes are derived from the actual output images.
        """
        if not self._record:
            self._writers_ready = True
            return

        os.makedirs(self._video_dir, exist_ok=True)

        ov_h,  ov_w  = overlay_frame.shape[:2]
        occ_h, occ_w = occ_frame.shape[:2]
        fps           = float(self.cam_fps)

        try:
            self._w_overlay = _open_mp4_writer(
                os.path.join(self._video_dir, 'lane_overlay.mp4'),
                ov_w, ov_h, fps)
            # debug mosaic is always exactly CANVAS_W × CANVAS_H
            self._w_debug = _open_mp4_writer(
                os.path.join(self._video_dir, 'debug_grid.mp4'),
                CANVAS_W, CANVAS_H, fps)
            self._w_occ = _open_mp4_writer(
                os.path.join(self._video_dir, 'occupancy_grid.mp4'),
                occ_w, occ_h, fps)
        except RuntimeError as e:
            self.get_logger().error(f'MP4 writer init failed: {e}')
            self._record = False
            self._writers_ready = True
            return

        self._h265_overlay_file = open(
            os.path.join(self._video_dir, 'lane_overlay.h265'), 'wb')
        self._h265_debug_file   = open(
            os.path.join(self._video_dir, 'debug_grid.h265'),   'wb')

        self._writers_ready = True
        self.get_logger().info(
            f'Recording started → {self._video_dir}\n'
            f'  lane_overlay.mp4    [{ov_w}x{ov_h}]\n'
            f'  debug_grid.mp4      [{CANVAS_W}x{CANVAS_H}]\n'
            f'  occupancy_grid.mp4  [{occ_w}x{occ_h}]\n'
            f'  lane_overlay.h265   (raw H.265)\n'
            f'  debug_grid.h265     (raw H.265)'
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Write frames
    # ──────────────────────────────────────────────────────────────────────

    def _write_frames(self,
                      overlay_frame: np.ndarray,
                      debug_frame:   np.ndarray,
                      occ_frame:     np.ndarray) -> None:
        if not self._record:
            return
        if self._w_overlay is not None:
            self._w_overlay.write(overlay_frame)
        if self._w_debug is not None:
            self._w_debug.write(debug_frame)
        if self._w_occ is not None:
            self._w_occ.write(occ_frame)

    def _write_h265(self) -> None:
        """Drain both H.265 encoder queues and append raw packets to files."""
        if not self._record:
            return
        if (self._h265_overlay_file is not None and
                self._enc_overlay_queue is not None):
            while self._enc_overlay_queue.has():
                pkt = self._enc_overlay_queue.get()
                if pkt is not None:
                    pkt.getData().tofile(self._h265_overlay_file)

        if (self._h265_debug_file is not None and
                self._enc_debug_queue is not None):
            while self._enc_debug_queue.has():
                pkt = self._enc_debug_queue.get()
                if pkt is not None:
                    pkt.getData().tofile(self._h265_debug_file)

    # ──────────────────────────────────────────────────────────────────────
    #  Timer callback
    # ──────────────────────────────────────────────────────────────────────

    def _timer_cb(self):
        if not self._pipeline.isRunning():
            return

        in_rgb = self._video_queue.get() \
            if self._video_queue.has() else None
        if in_rgb is None:
            return

        frame = in_rgb.getCvFrame()   # already 180°-rotated by ImageManip

        result, dbg, occ_display, occ_grid = self._detect_lane(frame)

        # ── Lazy writer init (first frame only) ───────────────────────────
        if not self._writers_ready:
            self._init_writers(result, occ_display)

        # ── Write MP4 frames ──────────────────────────────────────────────
        self._write_frames(result, dbg, occ_display)

        # ── Drain H.265 encoder queues ────────────────────────────────────
        self._write_h265()

        stamp = self.get_clock().now().to_msg()

        # ── Publish images ────────────────────────────────────────────────
        self.pub_result.publish(
            self.bridge.cv2_to_imgmsg(result,      encoding='bgr8'))
        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(dbg,         encoding='bgr8'))
        self.pub_occ_img.publish(
            self.bridge.cv2_to_imgmsg(occ_display, encoding='bgr8'))

        # ── Publish occupancy grid ────────────────────────────────────────
        self.pub_occ.publish(
            self._to_ros_occ_grid(occ_grid, stamp))

        # ── Publish coefficients ──────────────────────────────────────────
        m   = self._metrics
        dim = MultiArrayDimension(label='lane_coeffs', size=6, stride=6)
        msg_c            = Float32MultiArray()
        msg_c.layout.dim = [dim]
        msg_c.data       = [
            float(m['y_A']), float(m['y_B']), float(m['y_bot']),
            float(m['d_A']), float(m['d_B']), 0.0,
        ]
        self.pub_coeffs.publish(msg_c)

        # ── Publish status JSON ───────────────────────────────────────────
        s_msg      = String()
        s_msg.data = (
            f'{{'
            f'"status":"{m["status"]}",'
            f'"frame":{m["frame"]},'
            f'"y_mode":"{m["y_mode"]}",'
            f'"pts_yellow":{m["pts_yellow"]},'
            f'"y_recovery":{str(m["y_recovery"]).lower()},'
            f'"y_A":{m["y_A"]:.6f},'
            f'"y_B":{m["y_B"]:.4f},'
            f'"y_bot":{m["y_bot"]:.2f},'
            f'"y_lat":{m["y_lat"]:.3f},'
            f'"y_curv":{m["y_curv"]:.5f},'
            f'"d_mode":"{m["d_mode"]}",'
            f'"pts_dash":{m["pts_dash"]},'
            f'"d_recovery":{str(m["d_recovery"]).lower()},'
            f'"d_A":{m["d_A"]:.6f},'
            f'"d_B":{m["d_B"]:.4f},'
            f'"d_lat":{m["d_lat"]:.3f},'
            f'"d_curv":{m["d_curv"]:.5f},'
            f'"l2c":{self.l2c},'
            f'"c2r":{self.c2r}'
            f'}}'
        )
        self.pub_status.publish(s_msg)

        # ── Optional local windows ────────────────────────────────────────
        if self.show_win:
            cv2.imshow('sumit_laneCV | Final',
                       cv2.resize(result, (1280, 720)))
            cv2.imshow('sumit_laneCV | Debug',
                       cv2.resize(dbg, (1280, 720)))
            cv2.imshow('sumit_laneCV | Occupancy', occ_display)
            cv2.waitKey(1)

    # ──────────────────────────────────────────────────────────────────────
    #  OccupancyGrid → ROS message
    # ──────────────────────────────────────────────────────────────────────

    def _to_ros_occ_grid(self,
                         occ_grid: np.ndarray,
                         stamp) -> OccupancyGrid:
        msg                           = OccupancyGrid()
        msg.header.stamp              = stamp
        msg.header.frame_id           = 'base_footprint_imu'
        msg.info.resolution           = (
            float(UNWARPED_SIZE[0]) / OCC_COLS / 1000.0)
        msg.info.width                = OCC_COLS
        msg.info.height               = OCC_ROWS
        msg.info.origin               = Pose()
        msg.info.origin.orientation.w = 1.0

        ros = np.full((OCC_ROWS, OCC_COLS), -1, dtype=np.int8)
        ros[occ_grid == OCC_FREE]     = 0
        ros[occ_grid == OCC_OCCUPIED] = 100
        msg.data = ros[::-1].flatten().tolist()
        return msg

    # ──────────────────────────────────────────────────────────────────────
    #  MAIN DETECTION PIPELINE
    # ──────────────────────────────────────────────────────────────────────

    def _detect_lane(self, frame: np.ndarray):
        self._frame_idx += 1

        # ── Resize / undistort / warp ─────────────────────────────────────
        if (frame.shape[1], frame.shape[0]) != ORIGINAL_SIZE:
            frame = cv2.resize(frame, ORIGINAL_SIZE)
        undist = cv2.undistort(frame, K, D)
        warped = cv2.warpPerspective(
            undist, M, UNWARPED_SIZE, flags=cv2.INTER_LINEAR)
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # ══════════════════════════════════════════════════════════════════
        #  YELLOW PIPELINE  (v2)
        # ══════════════════════════════════════════════════════════════════
        yellow_mask = yellow_extract_mask(warped)
        yellow_roi  = yellow_mask.copy()
        yellow_roi[:ROI_TOP, :] = 0
        yellow_filt = yellow_filter_contours(yellow_roi)

        (y_coeffs_new,
         ys_y, xs_y,
         y_bot) = yellow_fit(yellow_filt, self._y_last_x, self._y_recovery)

        y_lat = y_curv = 0.0
        y_mode = 'DETECT'

        if y_coeffs_new is not None:
            valid, y_lat, y_curv = yellow_is_valid(
                y_coeffs_new, self._y_history)
            if valid or self._y_recovery:
                self._y_history.append(y_coeffs_new)
                self._y_last_x   = y_bot
                self._y_rejects  = 0
                self._y_recovery = False
            else:
                self._y_rejects += 1
                if self._y_rejects >= Y_RECOVERY_THR:
                    self._y_recovery = True
                    self._y_last_x   = None
                y_mode = 'REJ'
        else:
            self._y_rejects += 1
            if self._y_rejects >= Y_RECOVERY_THR:
                self._y_recovery = True
                self._y_last_x   = None
            y_mode = 'LOW_PTS'

        left_coeffs = (np.mean(self._y_history, axis=0)
                       if self._y_history else None)
        pts_yellow  = len(ys_y)

        # ══════════════════════════════════════════════════════════════════
        #  DASH PIPELINE  (v1)
        # ══════════════════════════════════════════════════════════════════
        _, edges_gated, _ = dash_preprocess(gray)

        dash_roi = edges_gated.copy()
        dash_roi[:ROI_TOP, :]   = 0
        dash_roi[:, :ROI_LEFT]  = 0
        dash_roi[:, ROI_RIGHT:] = 0

        dash_filt = dash_filter_contours(dash_roi)

        self._d_last_cx = dash_find_center_x(
            dash_filt, self._d_last_cx, self._d_recovery)

        d_coeffs_new, ys_d, xs_d = dash_fit(
            dash_filt, self._d_last_cx, self._d_recovery)

        d_lat = d_curv = 0.0
        d_mode = 'DETECT'

        if d_coeffs_new is not None:
            valid, d_lat, d_curv = dash_is_valid(
                d_coeffs_new, self._d_history)
            if valid or self._d_recovery:
                self._d_history.append(d_coeffs_new)
                self._d_rejects        = 0
                self._d_consec_rejects = 0          # reset on accept
                self._d_recovery       = False
            else:
                self._d_rejects        += 1
                self._d_consec_rejects += 1
                if self._d_rejects >= D_RECOVERY_THR:
                    self._d_recovery       = True
                    self._d_last_cx        = None
                    self._d_consec_rejects = 0      # reset on recovery entry
                    self._d_history.clear()
                d_mode = 'REJ'
        else:
            self._d_rejects        += 1
            self._d_consec_rejects += 1
            if self._d_rejects >= D_RECOVERY_THR:
                self._d_recovery       = True
                self._d_last_cx        = None
                self._d_consec_rejects = 0          # reset on recovery entry
                self._d_history.clear()
            d_mode = 'LOW_PTS'

        # Use smoothed dash history only if not lost too long
        if (self._d_history and
                self._d_consec_rejects < D_OFFSET_FALLBACK_THR):
            raw_center = np.mean(self._d_history, axis=0)
        else:
            raw_center = None
            if d_mode == 'DETECT':
                d_mode = 'OFFSET'

        pts_dash = len(ys_d)

        # ── Candidate center: detected OR yellow+offset ───────────────────
        if raw_center is not None:
            candidate = raw_center
        elif left_coeffs is not None:
            candidate    = left_coeffs.copy()
            candidate[2] = left_coeffs[2] + self.l2c
            d_mode       = 'OFFSET'
        else:
            candidate = None

        # ── Output smoother — single lateral-jump gate ────────────────────
        if candidate is not None:
            if len(self._center_out_history) > 0:
                last_out = np.mean(self._center_out_history, axis=0)
                jump     = abs(
                    np.polyval(candidate, UNWARPED_SIZE[1] - 1) -
                    np.polyval(last_out,  UNWARPED_SIZE[1] - 1))
                if jump <= D_OUTPUT_MAX_LAT:
                    self._center_out_history.append(candidate)
                # else: hold last output — do not update history this frame
            else:
                self._center_out_history.append(candidate)  # first frame

        center_coeffs = (np.mean(self._center_out_history, axis=0)
                         if self._center_out_history else None)

        # ══════════════════════════════════════════════════════════════════
        #  RIGHT EDGE + SAMPLE CURVES
        # ══════════════════════════════════════════════════════════════════
        ploty = np.linspace(ROI_TOP, UNWARPED_SIZE[1] - 1, 200)

        left_x   = (np.polyval(left_coeffs, ploty)
                    if left_coeffs   is not None
                    else np.full_like(ploty, 0.0))
        center_x = (np.polyval(center_coeffs, ploty)
                    if center_coeffs is not None
                    else left_x + self.l2c)
        right_x  = center_x + self.c2r

        l_valid = (left_x   >= 0) & (left_x   < UNWARPED_SIZE[0])
        c_valid = (center_x >= 0) & (center_x < UNWARPED_SIZE[0])
        r_valid = (right_x  >= 0) & (right_x  < UNWARPED_SIZE[0])

        # ══════════════════════════════════════════════════════════════════
        #  METRICS — update once, before any rendering
        # ══════════════════════════════════════════════════════════════════
        status = 'OK' if left_coeffs is not None else 'NO_YELLOW'

        self._metrics.update({
            'status':     status,
            'frame':      self._frame_idx,
            'l2c':        self.l2c,
            'c2r':        self.c2r,
            'y_mode':     y_mode,
            'pts_yellow': pts_yellow,
            'y_recovery': self._y_recovery,
            'y_bot':      (float(np.polyval(left_coeffs, UNWARPED_SIZE[1] - 1))
                           if left_coeffs is not None else 0.0),
            'y_A': float(left_coeffs[0])   if left_coeffs   is not None else 0.0,
            'y_B': float(left_coeffs[1])   if left_coeffs   is not None else 0.0,
            'y_lat':  y_lat,
            'y_curv': y_curv,
            'd_mode':     d_mode,
            'pts_dash':   pts_dash,
            'd_recovery': self._d_recovery,
            'd_A': float(center_coeffs[0]) if center_coeffs is not None else 0.0,
            'd_B': float(center_coeffs[1]) if center_coeffs is not None else 0.0,
            'd_lat':  d_lat,
            'd_curv': d_curv,
        })

        # ══════════════════════════════════════════════════════════════════
        #  OCCUPANCY GRID
        # ══════════════════════════════════════════════════════════════════
        occ_grid    = build_occupancy_grid(left_coeffs, right_x, ploty)
        occ_display = render_occupancy_grid(occ_grid, self._metrics)

        # ══════════════════════════════════════════════════════════════════
        #  FIT VISUALISATION
        # ══════════════════════════════════════════════════════════════════
        fit_vis = build_fit_vis(
            ys_y, xs_y, ys_d, xs_d,
            left_x, center_x, right_x,
            ploty, l_valid, c_valid, r_valid)

        # ══════════════════════════════════════════════════════════════════
        #  POLYGON OVERLAY → UNWARP → RESULT
        # ══════════════════════════════════════════════════════════════════
        bird_ov = np.zeros(
            (UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)

        pts_l = np.array([[int(left_x[i]),   int(ploty[i])]
                          for i in range(len(ploty)) if l_valid[i]],
                         dtype=np.int32)
        pts_c = np.array([[int(center_x[i]), int(ploty[i])]
                          for i in range(len(ploty)) if c_valid[i]],
                         dtype=np.int32)
        pts_r = np.array([[int(right_x[i]),  int(ploty[i])]
                          for i in range(len(ploty)) if r_valid[i]],
                         dtype=np.int32)

        if len(pts_l) > 2 and len(pts_r) > 2:
            cv2.fillPoly(bird_ov,
                         [np.concatenate([pts_l, np.flipud(pts_r)])],
                         (0, 180, 0))
        if len(pts_l) > 1:
            cv2.polylines(bird_ov, [pts_l], False, (0, 215, 255), 5)
        if len(pts_c) > 1:
            cv2.polylines(bird_ov, [pts_c], False, (255, 255,   0), 3)
        if len(pts_r) > 1:
            cv2.polylines(bird_ov, [pts_r], False, (0,   0, 255), 3)

        overlay_inv = cv2.warpPerspective(
            bird_ov, M_inv, ORIGINAL_SIZE, flags=cv2.INTER_LINEAR)
        result = cv2.addWeighted(undist, 1.0, overlay_inv, 0.5, 0)
        result = draw_metrics_panel(result, self._metrics)

        # ══════════════════════════════════════════════════════════════════
        #  DEBUG MOSAIC  (1920×1080, 7-panel)
        # ══════════════════════════════════════════════════════════════════
        dbg = build_debug_mosaic_1080p(
            undist        = undist,
            warped        = warped,
            yellow_filt   = yellow_filt,
            dash_filt     = dash_filt,
            fit_vis       = fit_vis,
            occ_grid      = occ_grid,
            overlay_frame = result,
            metrics       = self._metrics
        )

        return result, dbg, occ_display, occ_grid

    # ──────────────────────────────────────────────────────────────────────
    #  Cleanup
    # ──────────────────────────────────────────────────────────────────────

    def destroy_node(self):
        for attr, label in (('_w_overlay', 'lane_overlay.mp4'),
                            ('_w_debug',   'debug_grid.mp4'),
                            ('_w_occ',     'occupancy_grid.mp4')):
            w = getattr(self, attr, None)
            if w is not None:
                w.release()
                self.get_logger().info(f'Saved {label}')

        self._write_h265()
        for attr, label in (('_h265_overlay_file', 'lane_overlay.h265'),
                            ('_h265_debug_file',   'debug_grid.h265')):
            f = getattr(self, attr, None)
            if f is not None:
                f.close()
                self.get_logger().info(f'Saved {label}')

        if self.show_win:
            cv2.destroyAllWindows()

        super().destroy_node()


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main(args=None):
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