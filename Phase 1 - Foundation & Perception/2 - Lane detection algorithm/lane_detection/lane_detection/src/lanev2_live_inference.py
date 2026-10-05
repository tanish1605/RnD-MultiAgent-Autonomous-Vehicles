#!/usr/bin/env python3
"""
Live Lane Detection CV Pipeline — OAK-D Camera
================================================
File   : lane_cv_oakd_live.py
Author : Sumit
System : Ubuntu 22.04 | DepthAI 3.6.1 API v3 | OpenCV 4.x | NumPy

Captures live feed from an OAK-D camera (mounted 180° — feed is rotated),
runs the full lane detection CV pipeline and saves 3 output videos:
    1. debug_grid.mp4       — 8-panel debug mosaic
    2. lane_overlay.mp4     — Polygon overlay on road video
    3. occupancy_grid.mp4   — Occupancy grid only

Lane Detection Strategy:
    PRIMARY   : Yellow tape on left edge  (HSV color mask)
    SECONDARY : Center dashed lines       (adaptive threshold + dash filter)
                When center dashes missing → estimated from yellow + offset
    RIGHT     : Always offset-based       (no CV tracking)

Usage:
    python lane_cv_oakd_live.py
    python lane_cv_oakd_live.py --no-display
    python lane_cv_oakd_live.py --fps 15 --duration 60
    python lane_cv_oakd_live.py --out-debug my_debug.mp4 \
                                 --out-overlay my_overlay.mp4 \
                                 --out-occ my_occ.mp4
"""

import argparse
import sys
import signal
import time
import cv2
import numpy as np
import depthai as dai
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS / CONFIG
# ═══════════════════════════════════════════════════════════════════════════

UNWARPED_SIZE = (500, 600)      # (width, height) bird's-eye canvas
ORIGINAL_SIZE = (1920, 1080)    # camera output resolution

# ── Perspective warp control points ──────────────────────────────────────
# NOTE: These are defined for the CORRECTED (post-180° rotation) image.
#       If your warp quad was tuned on the un-rotated feed, flip the points:
#           SRC_rotated[i] = (W-1 - SRC_orig[i][0], H-1 - SRC_orig[i][1])
#       The values below already assume the rotated view.
SRC = np.array(
    [(743, 413), (1237, 419), (1904, 806), (10, 738)],
    dtype=np.float32
)
DST = np.array(
    [(0, 0), (500, 0), (500, 600), (0, 600)],
    dtype=np.float32
)

# ── Camera intrinsics (OAK-D fisheye lens) ───────────────────────────────
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

# ── Pre-compute perspective transforms ───────────────────────────────────
M     = cv2.getPerspectiveTransform(SRC, DST)
M_inv = cv2.getPerspectiveTransform(DST, SRC)

# ══════════════════════════════════════════════════════════════════════════
#  YELLOW TAPE CONFIG  (PRIMARY — left edge)
# ══════════════════════════════════════════════════════════════════════════
YELLOW_HSV_LOW  = np.array([ 18,  80,  80], dtype=np.uint8)
YELLOW_HSV_HIGH = np.array([ 40, 255, 255], dtype=np.uint8)

YELLOW_OPEN_K   = (5,  5)
YELLOW_CLOSE_K  = (15, 15)
YELLOW_MIN_AREA = 200
YELLOW_MIN_PTS  = 25

# ══════════════════════════════════════════════════════════════════════════
#  CENTER DASHED LINE CONFIG  (SECONDARY)
# ══════════════════════════════════════════════════════════════════════════
CANNY_LOW   = 255
CANNY_HIGH  = 255
CLAHE_CLIP  = 1.5
CLAHE_GRID  = (8, 8)
ADAPT_BLOCK = 199
ADAPT_C     = -45

MIN_DASH_AREA     = 120
MAX_DASH_AREA     = 2000
MIN_DASH_ASPECT   = 1.4
MIN_DASH_SOLIDITY = 0.7
CENTER_MIN_PTS    = 15

# ══════════════════════════════════════════════════════════════════════════
#  LANE GEOMETRY OFFSETS  (bird's-eye pixel space)
# ══════════════════════════════════════════════════════════════════════════
LEFT_TO_CENTER_OFFSET  = 230
CENTER_TO_RIGHT_OFFSET = 230

# ══════════════════════════════════════════════════════════════════════════
#  ROI  (bird's-eye pixel space)
# ══════════════════════════════════════════════════════════════════════════
ROI_TOP   = 280
ROI_LEFT  = 0
ROI_RIGHT = 485

# ══════════════════════════════════════════════════════════════════════════
#  TRACKING / SMOOTHING
# ══════════════════════════════════════════════════════════════════════════
MAX_LATERAL_JUMP              = 150
MAX_CURV_JUMP                 = 0.15
SMOOTH_N                      = 5
RECOVERY_THRESHOLD            = 8
YELLOW_SEARCH_WINDOW          = 120
YELLOW_SEARCH_WINDOW_RECOVERY = 400

# ══════════════════════════════════════════════════════════════════════════
#  OCCUPANCY GRID
# ══════════════════════════════════════════════════════════════════════════
OCC_COLS      = 100
OCC_ROWS      = 120
OCC_DISPLAY_W = 400
OCC_DISPLAY_H = 480

OCC_FREE     = 255
OCC_UNKNOWN  = 128
OCC_OCCUPIED = 0

# ══════════════════════════════════════════════════════════════════════════
#  DEBUG TILE SIZE
# ══════════════════════════════════════════════════════════════════════════
DEBUG_W = 480
DEBUG_H = 360

# ── Pre-compute CLAHE ────────────────────────────────────────────────────
CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — YELLOW TAPE  (PRIMARY)
# ═══════════════════════════════════════════════════════════════════════════

def extract_yellow_mask(bgr_warped: np.ndarray) -> np.ndarray:
    """HSV colour threshold → bright yellow tape mask."""
    hsv  = cv2.cvtColor(bgr_warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_HSV_LOW, YELLOW_HSV_HIGH)
    open_k  = cv2.getStructuringElement(cv2.MORPH_RECT, YELLOW_OPEN_K)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, YELLOW_CLOSE_K)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    return mask


def filter_yellow_contours(yellow_mask: np.ndarray) -> np.ndarray:
    """Keep only contours >= YELLOW_MIN_AREA; return filtered binary mask."""
    filtered = np.zeros_like(yellow_mask)
    contours, _ = cv2.findContours(
        yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= YELLOW_MIN_AREA:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
    return filtered


def fit_yellow_left(yellow_filtered: np.ndarray,
                    last_left_x,
                    recovery: bool):
    """
    2nd-degree polynomial fit to yellow tape pixels.

    Returns (coeffs | None, ys, xs, left_x_bottom)
    """
    hist   = np.sum(yellow_filtered, axis=0).astype(np.float32)
    window = (YELLOW_SEARCH_WINDOW_RECOVERY
              if recovery else YELLOW_SEARCH_WINDOW)

    if last_left_x is not None:
        lo = int(max(0, last_left_x - window))
        hi = int(min(UNWARPED_SIZE[0], last_left_x + window))
        windowed = np.zeros_like(hist)
        windowed[lo:hi] = hist[lo:hi]
        if windowed.max() >= 5:
            hist = windowed

    if hist.max() < 5:
        return None, np.array([]), np.array([]), last_left_x

    peak_x = int(np.argmax(hist))
    lo     = max(0, peak_x - window)
    hi     = min(UNWARPED_SIZE[0], peak_x + window)
    locked = np.zeros_like(yellow_filtered)
    locked[:, lo:hi] = yellow_filtered[:, lo:hi]

    ys, xs = np.where(locked > 0)
    if len(ys) < YELLOW_MIN_PTS:
        return None, ys, xs, last_left_x

    coeffs        = np.polyfit(ys, xs, 2)
    left_x_bottom = float(np.polyval(coeffs, UNWARPED_SIZE[1] - 1))
    return coeffs, ys, xs, left_x_bottom


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — CENTER DASHES  (SECONDARY)
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_center_dashes(gray: np.ndarray):
    """CLAHE → adaptive threshold → Canny → gated edges."""
    enhanced  = CLAHE.apply(gray)
    tape_mask = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        ADAPT_BLOCK, ADAPT_C
    )
    open_k    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_OPEN, open_k)
    blurred   = cv2.GaussianBlur(enhanced, (5, 5), 0)
    canny     = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    edges_gated = cv2.bitwise_and(canny, tape_mask)
    return enhanced, edges_gated, tape_mask


def filter_dashes(roi_mask: np.ndarray) -> np.ndarray:
    """Morphological close + contour filter for centre dash segments."""
    close_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed   = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, close_k)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated  = cv2.dilate(closed, dilate_k, iterations=2)
    filtered = np.zeros_like(dilated)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_DASH_AREA <= area <= MAX_DASH_AREA):
            continue
        _, _, w, h = cv2.boundingRect(cnt)
        if (h / (w + 1e-5)) < MIN_DASH_ASPECT:
            continue
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        if (area / hull_area) < MIN_DASH_SOLIDITY:
            continue
        cv2.drawContours(filtered, [cnt], -1, 255, -1)
    return filtered


# ═══════════════════════════════════════════════════════════════════════════
#  VALIDATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def is_valid_fit(new_coeffs: np.ndarray, history: deque):
    """Reject large lateral or curvature jumps. Returns (valid, lat, curv)."""
    if len(history) == 0:
        return True, 0.0, 0.0
    last      = np.mean(history, axis=0)
    lat_jump  = abs(np.polyval(new_coeffs, UNWARPED_SIZE[1] - 1)
                    - np.polyval(last,     UNWARPED_SIZE[1] - 1))
    curv_jump = abs(new_coeffs[0] - last[0])
    valid     = (lat_jump <= MAX_LATERAL_JUMP) and (curv_jump <= MAX_CURV_JUMP)
    return valid, float(lat_jump), float(curv_jump)


# ═══════════════════════════════════════════════════════════════════════════
#  OCCUPANCY GRID
# ═══════════════════════════════════════════════════════════════════════════

def build_occupancy_grid(left_coeffs,
                         right_x_arr: np.ndarray,
                         ploty:       np.ndarray) -> np.ndarray:
    """
    uint8 occupancy grid.
    OCC_FREE=255  OCC_OCCUPIED=0  OCC_UNKNOWN=128
    """
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
                OCC_FREE if lx <= bird_x <= rx else OCC_OCCUPIED
            )

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
    """Render uint8 occupancy grid as labelled BGR display image."""
    h, w   = grid.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    colour[grid == OCC_FREE]     = (255, 255, 255)
    colour[grid == OCC_UNKNOWN]  = (30,  30,  30)
    colour[grid == OCC_OCCUPIED] = (0,   0,   0)

    disp = cv2.resize(colour, (OCC_DISPLAY_W, OCC_DISPLAY_H),
                      interpolation=cv2.INTER_NEAREST)

    cell_w     = OCC_DISPLAY_W / OCC_COLS
    cell_h     = OCC_DISPLAY_H / OCC_ROWS
    grid_color = (60, 60, 60)
    for c in range(OCC_COLS + 1):
        x = int(c * cell_w)
        cv2.line(disp, (x, 0), (x, OCC_DISPLAY_H), grid_color, 1)
    for r in range(OCC_ROWS + 1):
        y = int(r * cell_h)
        cv2.line(disp, (0, y), (OCC_DISPLAY_W, y), grid_color, 1)

    legend = np.zeros((140, OCC_DISPLAY_W, 3), dtype=np.uint8)

    def swatch(x: int, y: int, color: tuple, label: str):
        cv2.rectangle(legend, (x, y), (x + 18, y + 14), color, -1)
        if color == (255, 255, 255):
            cv2.rectangle(legend, (x, y), (x + 18, y + 14),
                          (100, 100, 100), 1)
        cv2.putText(legend, label, (x + 22, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

    swatch(10,  6, (255, 255, 255), 'Free / driveable')
    swatch(10, 24, (0,   0,   0),   'Boundary / occupied')
    swatch(10, 42, (30,  30,  30),  'Unknown / outside ROI')

    sc = ((0, 220, 0)
          if metrics['status'].startswith('OK')
          else (0, 100, 255))

    cv2.putText(legend,
                f"Status : {metrics['status']}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sc, 1)
    cv2.putText(legend,
                f"Left: {metrics['center_mode_left']}  "
                f"Center: {metrics['center_mode']}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 180, 180), 1)
    cv2.putText(legend,
                f"Yellow pts: {metrics['pts_yellow']:3d}   "
                f"Dash pts: {metrics['pts_center']:3d}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 180, 180), 1)
    cv2.putText(legend,
                f"Grid {OCC_COLS}x{OCC_ROWS}  "
                f"({UNWARPED_SIZE[0]/OCC_COLS:.1f}x"
                f"{UNWARPED_SIZE[1]/OCC_ROWS:.1f} px/cell)",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (140, 140, 140), 1)

    title = np.zeros((30, OCC_DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(title, 'OCCUPANCY GRID  |  Yellow-Primary Pipeline',
                (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    return np.vstack([title, disp, legend])


# ═══════════════════════════════════════════════════════════════════════════
#  DEBUG / VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a label bar into the top-left of the image."""
    out   = img.copy()
    bar_w = min(len(text) * 10 + 14, img.shape[1])
    cv2.rectangle(out, (0, 0), (bar_w, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    return out


def to_tile(img: np.ndarray,
            w: int = DEBUG_W,
            h: int = DEBUG_H) -> np.ndarray:
    """Resize to tile dimensions; convert grayscale → BGR if needed."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (w, h))


def draw_roi_on_warped(warped: np.ndarray) -> np.ndarray:
    """Draw the active ROI rectangle on the bird's-eye image."""
    vis = warped.copy()
    cv2.rectangle(vis,
                  (ROI_LEFT, ROI_TOP),
                  (ROI_RIGHT, UNWARPED_SIZE[1] - 1),
                  (0, 255, 255), 2)
    cv2.putText(vis, 'ROI', (ROI_LEFT + 4, ROI_TOP + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return vis


def draw_metrics_panel(img: np.ndarray, m: dict) -> np.ndarray:
    """Semi-transparent metrics overlay on the annotated frame."""
    panel   = img.copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (520, 360), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)

    C_OK   = (0, 255, 0)
    C_WARN = (0, 165, 255)
    C_BAD  = (0, 0, 255)
    C_INFO = (255, 255, 255)
    C_YEL  = (0, 220, 220)

    def put(text: str, y: int, color: tuple = C_INFO):
        cv2.putText(panel, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1)

    sc = (C_OK   if m['status'].startswith('OK')  else
          C_WARN if m['status'].startswith('REJ') else C_BAD)

    put(f"Status      : {m['status']}",  28, sc)
    put(f"Recovery    : {'YES' if m['recovery'] else 'no'}", 52,
        C_WARN if m['recovery'] else C_OK)
    put(f"[YELLOW] pts: {m['pts_yellow']:3d}  "
        f"mode: {m['center_mode_left']}",
        76,
        C_OK if m['pts_yellow'] >= YELLOW_MIN_PTS else C_BAD)
    put(f"[YELLOW] A={m['A_left']:+.6f}  B={m['B_left']:+.4f}", 100, C_YEL)
    put(f"[YELLOW] left_x_bot = {m['left_x_bot']:.1f} px", 124, C_YEL)
    put(f"[CENTER] pts: {m['pts_center']:3d}  "
        f"mode: {m['center_mode']}",
        148,
        C_OK if m['center_mode'] == 'DETECT' else C_WARN)
    put(f"[CENTER] A={m['A_center']:+.6f}  B={m['B_center']:+.4f}", 172, C_INFO)
    put(f"[RIGHT]  always offset ({CENTER_TO_RIGHT_OFFSET}px from center)",
        196, C_INFO)
    put(f"Lat jump  : {m['lateral_jump']:6.2f} px  "
        f"(max {MAX_LATERAL_JUMP})",
        220,
        C_OK if m['lateral_jump'] < MAX_LATERAL_JUMP else C_BAD)
    put(f"Curv jump : {m['curv_jump']:.5f}  (max {MAX_CURV_JUMP})",
        244,
        C_OK if m['curv_jump'] < MAX_CURV_JUMP else C_BAD)
    put(f"CLAHE={CLAHE_CLIP}  Canny={CANNY_LOW}/{CANNY_HIGH}  "
        f"AdaptB={ADAPT_BLOCK}", 268, C_INFO)
    put(f"L→C={LEFT_TO_CENTER_OFFSET}px  C→R={CENTER_TO_RIGHT_OFFSET}px",
        292, C_INFO)
    put(f"Frame: {m.get('frame', 0):6d}  "
        f"FPS: {m.get('fps', 0.0):.1f}", 316, C_INFO)
    put(f"Camera: 180deg rotated (corrected)", 340, (160, 160, 160))

    return panel


def build_poly_fit_vis(ys_y:     np.ndarray,
                       xs_y:     np.ndarray,
                       ys_d:     np.ndarray,
                       xs_d:     np.ndarray,
                       left_x:   np.ndarray,
                       center_x: np.ndarray,
                       right_x:  np.ndarray,
                       ploty:    np.ndarray,
                       l_valid:  np.ndarray,
                       c_valid:  np.ndarray,
                       r_valid:  np.ndarray) -> np.ndarray:
    """
    Black canvas showing raw pixel hits and three fitted polynomial curves.
    """
    fit_vis = np.zeros(
        (UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)

    if len(ys_y) > 0:
        # Clip coordinates to image bounds before indexing
        valid_mask = (
            (ys_y >= 0) & (ys_y < UNWARPED_SIZE[1]) &
            (xs_y >= 0) & (xs_y < UNWARPED_SIZE[0])
        )
        fit_vis[ys_y[valid_mask], xs_y[valid_mask]] = (0, 215, 255)

    if len(ys_d) > 0:
        valid_mask = (
            (ys_d >= 0) & (ys_d < UNWARPED_SIZE[1]) &
            (xs_d >= 0) & (xs_d < UNWARPED_SIZE[0])
        )
        fit_vis[ys_d[valid_mask], xs_d[valid_mask]] = (200, 200, 200)

    for i in range(len(ploty)):
        cy = int(ploty[i])
        if not (0 <= cy < UNWARPED_SIZE[1]):
            continue
        if l_valid[i]:
            lxi = int(left_x[i])
            if 0 <= lxi < UNWARPED_SIZE[0]:
                cv2.circle(fit_vis, (lxi, cy), 2, (0, 215, 255), -1)
        if c_valid[i]:
            cxi = int(center_x[i])
            if 0 <= cxi < UNWARPED_SIZE[0]:
                cv2.circle(fit_vis, (cxi, cy), 2, (255, 255, 0), -1)
        if r_valid[i]:
            rxi = int(right_x[i])
            if 0 <= rxi < UNWARPED_SIZE[0]:
                cv2.circle(fit_vis, (rxi, cy), 2, (0, 0, 255), -1)

    cv2.putText(fit_vis, 'Y=left  C=center  R=right',
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return fit_vis


def build_debug_mosaic(undist:        np.ndarray,
                       warped:        np.ndarray,
                       yellow_mask:   np.ndarray,
                       yellow_filt:   np.ndarray,
                       dash_filtered: np.ndarray,
                       fit_vis:       np.ndarray,
                       overlay_frame: np.ndarray,
                       occ_display:   np.ndarray) -> np.ndarray:
    """
    8-panel 4×2 debug mosaic:
    ┌──────────┬──────────┬──────────┬──────────┐
    │ 1 Undist │ 2 BirdEye│ 3 YelHSV │ 4 YelFlt │
    ├──────────┼──────────┼──────────┼──────────┤
    │ 5 DashFlt│ 6 PolyFit│ 7 Overlay│ 8 OccGrid│
    └──────────┴──────────┴──────────┴──────────┘
    """
    warped_roi = draw_roi_on_warped(warped)

    t1 = add_label(to_tile(undist,        DEBUG_W, DEBUG_H), "1: Undistorted (rot. corrected)")
    t2 = add_label(to_tile(warped_roi,    DEBUG_W, DEBUG_H), "2: Bird's-Eye + ROI")
    t3 = add_label(to_tile(yellow_mask,   DEBUG_W, DEBUG_H), "3: Yellow HSV Mask")
    t4 = add_label(to_tile(yellow_filt,   DEBUG_W, DEBUG_H), "4: Yellow Filtered")
    t5 = add_label(to_tile(dash_filtered, DEBUG_W, DEBUG_H), "5: White Dash Filter")
    t6 = add_label(to_tile(fit_vis,       DEBUG_W, DEBUG_H), "6: Poly Fit (Y/C/R)")
    t7 = add_label(to_tile(overlay_frame, DEBUG_W, DEBUG_H), "7: Lane Overlay")
    t8 = add_label(
        cv2.resize(occ_display, (DEBUG_W, DEBUG_H)),
        "8: Occupancy Grid"
    )
    row1 = np.hstack([t1, t2, t3, t4])
    row2 = np.hstack([t5, t6, t7, t8])
    return np.vstack([row1, row2])


# ═══════════════════════════════════════════════════════════════════════════
#  STATEFUL LANE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class LaneDetector:
    """
    Full per-frame lane detection with temporal tracking.
    Call process_frame(bgr) → (overlay, debug_mosaic, occ_display, occ_grid)

    The incoming BGR frame must already be rotation-corrected before calling
    this method (i.e. rotate 180° for an inverted camera).
    """

    def __init__(self,
                 l2c_offset: int = LEFT_TO_CENTER_OFFSET,
                 c2r_offset: int = CENTER_TO_RIGHT_OFFSET):

        self.l2c_offset = l2c_offset
        self.c2r_offset = c2r_offset

        # Yellow left edge tracking
        self._left_history  = deque(maxlen=SMOOTH_N)
        self._last_left_x   = None
        self._left_rejects  = 0
        self._left_recovery = False

        # Center dash tracking
        self._center_history = deque(maxlen=SMOOTH_N)

        self._frame_idx = 0
        self.metrics    = {
            'status':           'INIT',
            'recovery':         False,
            'pts_yellow':       0,
            'pts_center':       0,
            'center_mode_left': 'INIT',
            'center_mode':      'INIT',
            'left_x_bot':       0.0,
            'A_left':           0.0,
            'B_left':           0.0,
            'A_center':         0.0,
            'B_center':         0.0,
            'lateral_jump':     0.0,
            'curv_jump':        0.0,
            'frame':            0,
            'fps':              0.0,
        }

    # ─────────────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray):
        """
        Run the full detection pipeline on one BGR frame.

        Parameters
        ----------
        frame : BGR image at ORIGINAL_SIZE (already rotation-corrected)

        Returns
        -------
        overlay_frame : annotated road view (ORIGINAL_SIZE, BGR)
        debug_mosaic  : 8-panel debug grid  (4*DEBUG_W x 2*DEBUG_H, BGR)
        occ_display   : occupancy render    (BGR)
        occ_grid      : uint8 grid          (OCC_ROWS x OCC_COLS)
        """
        self._frame_idx += 1

        # ── 1. Resize guard ──────────────────────────────────────────────
        if (frame.shape[1], frame.shape[0]) != ORIGINAL_SIZE:
            frame = cv2.resize(frame, ORIGINAL_SIZE)

        # ── 2. Undistort ─────────────────────────────────────────────────
        undist = cv2.undistort(frame, K, D)

        # ── 3. Bird's-eye warp ───────────────────────────────────────────
        warped = cv2.warpPerspective(
            undist, M, UNWARPED_SIZE, flags=cv2.INTER_LINEAR)

        # ── 4. YELLOW LEFT EDGE (PRIMARY) ────────────────────────────────
        yellow_mask = extract_yellow_mask(warped)

        yellow_roi = yellow_mask.copy()
        yellow_roi[:ROI_TOP, :] = 0

        yellow_filtered = filter_yellow_contours(yellow_roi)

        (left_coeffs_new,
         ys_y, xs_y,
         left_x_bottom) = fit_yellow_left(
             yellow_filtered,
             self._last_left_x,
             self._left_recovery
         )

        lat_jump  = 0.0
        curv_jump = 0.0
        left_mode = 'DETECT'

        if left_coeffs_new is not None:
            valid, lat_jump, curv_jump = is_valid_fit(
                left_coeffs_new, self._left_history)

            if valid or self._left_recovery:
                self._left_history.append(left_coeffs_new)
                self._last_left_x   = left_x_bottom
                self._left_rejects  = 0
                self._left_recovery = False
            else:
                self._left_rejects += 1
                if self._left_rejects >= RECOVERY_THRESHOLD:
                    self._left_recovery = True
                    self._last_left_x   = None
                left_mode = 'REJ'
        else:
            self._left_rejects += 1
            if self._left_rejects >= RECOVERY_THRESHOLD:
                self._left_recovery = True
                self._last_left_x   = None
            left_mode = 'LOW_PTS'

        left_coeffs = (np.mean(self._left_history, axis=0)
                       if self._left_history else None)
        pts_yellow  = len(ys_y)

        # ── 5. CENTER DASHES (SECONDARY) ─────────────────────────────────
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, edges_gated, _ = preprocess_center_dashes(gray)

        dash_roi = edges_gated.copy()
        dash_roi[:ROI_TOP, :]   = 0
        dash_roi[:, :ROI_LEFT]  = 0
        dash_roi[:, ROI_RIGHT:] = 0

        dash_filtered = filter_dashes(dash_roi)
        ys_d, xs_d    = np.where(dash_filtered > 0)
        pts_center    = len(ys_d)

        # ── 6. CENTER LINE: DETECT or OFFSET ─────────────────────────────
        center_mode   = 'DETECT'
        center_coeffs = None

        if pts_center >= CENTER_MIN_PTS:
            new_center    = np.polyfit(ys_d, xs_d, 2)
            valid_c, _, _ = is_valid_fit(new_center, self._center_history)
            if valid_c:
                self._center_history.append(new_center)
            if self._center_history:
                center_coeffs = np.mean(self._center_history, axis=0)
                center_mode   = 'DETECT'
            else:
                center_mode   = 'OFFSET'
        else:
            center_mode = 'OFFSET'

        if center_mode == 'OFFSET' and left_coeffs is not None:
            center_coeffs    = left_coeffs.copy()
            center_coeffs[2] = left_coeffs[2] + self.l2c_offset

        # ── 7. RIGHT EDGE ─────────────────────────────────────────────────
        ploty = np.linspace(ROI_TOP, UNWARPED_SIZE[1] - 1, 200)

        left_x   = (np.polyval(left_coeffs,   ploty)
                    if left_coeffs   is not None
                    else np.full_like(ploty, 0.0))
        center_x = (np.polyval(center_coeffs, ploty)
                    if center_coeffs is not None
                    else left_x + self.l2c_offset)
        right_x  = center_x + self.c2r_offset

        l_valid = (left_x   >= 0) & (left_x   < UNWARPED_SIZE[0])
        c_valid = (center_x >= 0) & (center_x < UNWARPED_SIZE[0])
        r_valid = (right_x  >= 0) & (right_x  < UNWARPED_SIZE[0])

        # ── 8. Metrics ────────────────────────────────────────────────────
        status = (f'OK pts_y={pts_yellow} pts_c={pts_center}'
                  if left_coeffs is not None else 'NO_YELLOW')

        self.metrics.update({
            'status':           status,
            'recovery':         self._left_recovery,
            'pts_yellow':       pts_yellow,
            'pts_center':       pts_center,
            'center_mode_left': left_mode,
            'center_mode':      center_mode,
            'left_x_bot':       float(np.polyval(left_coeffs,
                                                 UNWARPED_SIZE[1] - 1))
                                if left_coeffs is not None else 0.0,
            'A_left':   float(left_coeffs[0])   if left_coeffs   is not None else 0.0,
            'B_left':   float(left_coeffs[1])   if left_coeffs   is not None else 0.0,
            'A_center': float(center_coeffs[0]) if center_coeffs is not None else 0.0,
            'B_center': float(center_coeffs[1]) if center_coeffs is not None else 0.0,
            'lateral_jump': lat_jump,
            'curv_jump':    curv_jump,
            'frame':        self._frame_idx,
        })

        # ── 9. Occupancy grid ─────────────────────────────────────────────
        occ_grid    = build_occupancy_grid(left_coeffs, right_x, ploty)
        occ_display = render_occupancy_grid(occ_grid, self.metrics)

        # ── 10. Poly-fit visualisation canvas ─────────────────────────────
        fit_vis = build_poly_fit_vis(
            ys_y, xs_y, ys_d, xs_d,
            left_x, center_x, right_x,
            ploty, l_valid, c_valid, r_valid
        )

        # ── 11. Colour overlay → unwarped back to original view ───────────
        overlay_bird = np.zeros(
            (UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)

        pts_l = np.array(
            [[int(left_x[i]),   int(ploty[i])]
             for i in range(len(ploty)) if l_valid[i]], dtype=np.int32)
        pts_c = np.array(
            [[int(center_x[i]), int(ploty[i])]
             for i in range(len(ploty)) if c_valid[i]], dtype=np.int32)
        pts_r = np.array(
            [[int(right_x[i]),  int(ploty[i])]
             for i in range(len(ploty)) if r_valid[i]], dtype=np.int32)

        if len(pts_l) > 2 and len(pts_r) > 2:
            fill = np.concatenate([pts_l, np.flipud(pts_r)], axis=0)
            cv2.fillPoly(overlay_bird, [fill], (0, 180, 0))

        if len(pts_l) > 1:
            cv2.polylines(overlay_bird, [pts_l], False, (0, 215, 255), 5)
        if len(pts_c) > 1:
            cv2.polylines(overlay_bird, [pts_c], False, (255, 255, 0), 3)
        if len(pts_r) > 1:
            cv2.polylines(overlay_bird, [pts_r], False, (0, 0, 255), 3)

        overlay_inv   = cv2.warpPerspective(
            overlay_bird, M_inv, ORIGINAL_SIZE, flags=cv2.INTER_LINEAR)
        overlay_frame = cv2.addWeighted(undist, 1.0, overlay_inv, 0.5, 0)
        overlay_frame = draw_metrics_panel(overlay_frame, self.metrics)

        # ── 12. Debug mosaic ──────────────────────────────────────────────
        debug_mosaic = build_debug_mosaic(
            undist, warped,
            yellow_mask, yellow_filtered,
            dash_filtered, fit_vis,
            overlay_frame, occ_display
        )

        return overlay_frame, debug_mosaic, occ_display, occ_grid


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO WRITER HELPER
# ═══════════════════════════════════════════════════════════════════════════

def open_writer(path: str,
                width: int,
                height: int,
                fps: float) -> cv2.VideoWriter:
    """Open an MP4 VideoWriter; raises RuntimeError on failure."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {path}")
    return writer


# ═══════════════════════════════════════════════════════════════════════════
#  DEPTHAI v3 PIPELINE  (follows the original ROS node's API exactly)
# ═══════════════════════════════════════════════════════════════════════════

def build_depthai_pipeline(fps: int):
    """
    Build and start a DepthAI 3.x (v3 API) pipeline.

    The 180° rotation is done in hardware via an ImageManip node so that
    every frame arriving from the queue is already orientation-correct —
    no cv2.rotate() call is needed in the main loop.

    Pipeline graph:
        Camera (CAM_A)
            └─ requestOutput ──► ImageManip (addRotateDeg 180)
                                      └─ manip.out ──► OutputQueue

    Uses:
        pipeline.create(dai.node.Camera).build(...)
        cam.requestOutput(...)
        pipeline.create(dai.node.ImageManip)
        manip.out.createOutputQueue(...)
        pipeline.start()

    Returns
    -------
    pipeline    : dai.Pipeline  (already started)
    video_queue : OutputQueue   — yields 180°-corrected BGR888p frames
    """
    pipeline = dai.Pipeline()

    # ── Camera node ───────────────────────────────────────────────────────
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A,
        sensorResolution=(ORIGINAL_SIZE[0], ORIGINAL_SIZE[1]),
        sensorFps=float(fps)
    )

    # ── Request raw output stream from the camera ─────────────────────────
    video_out = cam.requestOutput(
        size=(ORIGINAL_SIZE[0], ORIGINAL_SIZE[1]),
        type=dai.ImgFrame.Type.BGR888p
    )

    # ── ImageManip node — 180° rotation in hardware ───────────────────────
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.addRotateDeg(180)

    # Allocate enough frame buffer for a full BGR frame at ORIGINAL_SIZE.
    # Without this the node silently drops frames when resolution > 720p.
    max_bytes = ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1] * 3
    manip.setMaxOutputFrameSize(max_bytes)

    # Wire the camera output into the manip input
    video_out.link(manip.inputImage)

    # ── Output queue (v3 API) — reads from manip, not directly from camera ─
    video_queue = manip.out.createOutputQueue(
        maxSize=1,
        blocking=False
    )

    # ── Start the pipeline ────────────────────────────────────────────────
    pipeline.start()

    print(
        f"[DepthAI] OAK camera started  "
        f"({ORIGINAL_SIZE[0]}x{ORIGINAL_SIZE[1]} @ {fps} fps)  "
        f"API {dai.__version__}  |  "
        f"ImageManip: 180° hardware rotation active"
    )

    return pipeline, video_queue


# ═══════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Live OAK-D lane detection — saves 3 output videos')
    p.add_argument('--fps',        type=int,   default=30,
                   help='Camera capture FPS (default 30)')
    p.add_argument('--duration',   type=float, default=0,
                   help='Max recording duration in seconds  '
                        '(0 = run until Ctrl-C, default 0)')
    p.add_argument('--out-debug',   default='debug_grid2.mp4',
                   help='Output debug mosaic video  (default: debug_grid2.mp4)')
    p.add_argument('--out-overlay', default='lane_overlay2.mp4',
                   help='Output overlay video        (default: lane_overlay2.mp4)')
    p.add_argument('--out-occ',     default='occupancy_grid2.mp4',
                   help='Output occupancy grid video (default: occupancy_grid2.mp4)')
    p.add_argument('--no-display',  action='store_true',
                   help='Suppress live preview windows')
    p.add_argument('--l2c-offset',  type=int, default=LEFT_TO_CENTER_OFFSET,
                   help=f'Left→Centre offset px (default {LEFT_TO_CENTER_OFFSET})')
    p.add_argument('--c2r-offset',  type=int, default=CENTER_TO_RIGHT_OFFSET,
                   help=f'Centre→Right offset px (default {CENTER_TO_RIGHT_OFFSET})')
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── Graceful Ctrl-C handler ───────────────────────────────────────────
    _stop = [False]

    def _sig_handler(sig, frame):
        print("\n[INFO] Stop signal received — finishing up …")
        _stop[0] = True

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # ── Start DepthAI pipeline ────────────────────────────────────────────
    try:
        pipeline, video_queue = build_depthai_pipeline(args.fps)
    except Exception as e:
        print(f"[ERROR] DepthAI init failed: {e}")
        sys.exit(1)

    # ── Warm-up: grab one frame to get output image sizes ─────────────────
    print("[INFO] Waiting for first frame …", end='', flush=True)
    first_frame = None
    warmup_deadline = time.time() + 10.0          # 10-second timeout

    while first_frame is None and time.time() < warmup_deadline:
        if not pipeline.isRunning():
            print("\n[ERROR] Pipeline stopped during warm-up.")
            sys.exit(1)
        if video_queue.has():
            in_rgb      = video_queue.get()
            raw         = in_rgb.getCvFrame()
            first_frame = raw

        else:
            time.sleep(0.005)

    if first_frame is None:
        print("\n[ERROR] No frame received within 10 s — check camera.")
        sys.exit(1)
    print(" OK")

    # ── Initialise detector — process warm-up frame to get output sizes ───
    detector = LaneDetector(
        l2c_offset=args.l2c_offset,
        c2r_offset=args.c2r_offset
    )
    overlay_f, debug_f, occ_f, _ = detector.process_frame(first_frame.copy())

    ov_h,  ov_w  = overlay_f.shape[:2]
    dbg_h, dbg_w = debug_f.shape[:2]
    occ_h, occ_w = occ_f.shape[:2]

    # ── Open video writers ────────────────────────────────────────────────
    try:
        w_debug   = open_writer(args.out_debug,   dbg_w, dbg_h, float(args.fps))
        w_overlay = open_writer(args.out_overlay, ov_w,  ov_h,  float(args.fps))
        w_occ     = open_writer(args.out_occ,     occ_w, occ_h, float(args.fps))
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\n{'─'*62}")
    print(f"  Recording started")
    print(f"  Output 1 (debug)   : {args.out_debug}   [{dbg_w}x{dbg_h}]")
    print(f"  Output 2 (overlay) : {args.out_overlay} [{ov_w}x{ov_h}]")
    print(f"  Output 3 (occ)     : {args.out_occ}     [{occ_w}x{occ_h}]")
    if args.duration > 0:
        print(f"  Duration limit     : {args.duration:.1f} s")
    print(f"  Camera rotation    : 180°  (corrected in software)")
    print(f"  Press Ctrl-C to stop cleanly.")
    print(f"{'─'*62}\n")

    # Write warm-up frame
    w_debug.write(debug_f)
    w_overlay.write(overlay_f)
    w_occ.write(occ_f)

    # ── Preview windows ───────────────────────────────────────────────────
    SHOW = not args.no_display
    if SHOW:
        PREV_W = 1280
        PREV_H = int(dbg_h * PREV_W / max(dbg_w, 1))
        cv2.namedWindow('Debug Grid',     cv2.WINDOW_NORMAL)
        cv2.namedWindow('Lane Overlay',   cv2.WINDOW_NORMAL)
        cv2.namedWindow('Occupancy Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug Grid',    PREV_W, PREV_H)
        cv2.resizeWindow('Lane Overlay',  1280, 720)
        cv2.resizeWindow('Occupancy Grid', occ_w, occ_h)
        cv2.imshow('Debug Grid',     debug_f)
        cv2.imshow('Lane Overlay',   cv2.resize(overlay_f, (1280, 720)))
        cv2.imshow('Occupancy Grid', occ_f)
        cv2.waitKey(1)

    # ── FPS tracking ──────────────────────────────────────────────────────
    frame_count  = 1       # warm-up frame already counted
    t_start      = time.time()
    t_last_print = t_start
    fps_display  = float(args.fps)

    # ── Main capture loop ─────────────────────────────────────────────────
    while not _stop[0]:

        # Duration limit
        elapsed = time.time() - t_start
        if args.duration > 0 and elapsed >= args.duration:
            print(f"\n[INFO] Duration limit ({args.duration:.1f}s) reached.")
            break

        if not pipeline.isRunning():
            print("\n[WARN] DepthAI pipeline stopped unexpectedly.")
            break

        # Non-blocking queue poll
        if not video_queue.has():
            time.sleep(0.001)
            continue

        in_rgb = video_queue.get()
        raw    = in_rgb.getCvFrame()

        # ── Detection ─────────────────────────────────────────────────────
        overlay_f, debug_f, occ_f, _ = detector.process_frame(raw)

        # ── FPS calculation ───────────────────────────────────────────────
        frame_count += 1
        now          = time.time()
        dt           = now - t_last_print

        if dt >= 1.0:                               # update every second
            fps_display        = frame_count / (now - t_start)
            detector.metrics['fps'] = fps_display
            t_last_print       = now
            m = detector.metrics
            print(f"\r  Frame {frame_count:6d} | "
                  f"FPS {fps_display:5.1f} | "
                  f"Status: {m['status']:<35s} | "
                  f"Yellow: {m['pts_yellow']:3d}pts | "
                  f"Dashes: {m['pts_center']:3d}pts",
                  end='', flush=True)

        # ── Write frames ──────────────────────────────────────────────────
        w_debug.write(debug_f)
        w_overlay.write(overlay_f)
        w_occ.write(occ_f)

        # ── Live preview ──────────────────────────────────────────────────
        if SHOW:
            cv2.imshow('Debug Grid',     debug_f)
            cv2.imshow('Lane Overlay',   cv2.resize(overlay_f, (1280, 720)))
            cv2.imshow('Occupancy Grid', occ_f)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit key pressed — stopping.")
                break

    # ── Cleanup ───────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    avg_fps    = frame_count / max(total_time, 1e-6)

    print(f"\n\n{'─'*62}")
    print(f"  Recording finished")
    print(f"  Frames written : {frame_count}")
    print(f"  Duration       : {total_time:.1f} s")
    print(f"  Average FPS    : {avg_fps:.2f}")
    print(f"  Saved:")
    print(f"    {args.out_debug}")
    print(f"    {args.out_overlay}")
    print(f"    {args.out_occ}")
    print(f"{'─'*62}\n")

    w_debug.release()
    w_overlay.release()
    w_occ.release()

    if SHOW:
        cv2.destroyAllWindows()

    # Pipeline teardown — DepthAI v3 context handles device cleanup
    # No explicit stop call needed; garbage collection closes the device.
    print("[INFO] Done.")


if __name__ == '__main__':
    main()