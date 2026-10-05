#!/usr/bin/env python3
"""
Lane Detection v3 — Standalone Video Inference
===============================================
File   : lanev3_vid_inference.py
Author : Sumit
System : Ubuntu 22.04 | OpenCV 4.x | NumPy

Yellow tape  (LEFT)   : v2 HSV logic  — fully independent config + helpers
Center dashes (CENTER): v1 adaptive-threshold logic — fully independent config + helpers
Right edge            : always CENTER + C2R_OFFSET

Exports 3 MP4s:
    1. debug_grid.mp4
    2. lane_overlay.mp4
    3. occupancy_grid.mp4

Usage:
    python3 lanev3_vid_inference.py --input road.mp4
    python3 lanev3_vid_inference.py --input road.mp4 --no-display
"""

import argparse
import sys
import os
import cv2
import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
#  GEOMETRY — shared
# ═══════════════════════════════════════════════════════════════════════════

UNWARPED_SIZE = (500, 600)
ORIGINAL_SIZE = (1920, 1080)

SRC = np.array([
    (675, 543),
    (1247, 517),
    (1883, 851),
    (43,  882)
], dtype=np.float32)

DST = np.array(
    [(0, 0), (500, 0), (500, 600), (0, 600)],
    dtype=np.float32
)

K = np.array([
    [1459.2276611328125, 0.0, 945.8712768554688],
    [0.0, 1458.7711181640625, 516.719482421875],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

D = np.array([
    -2.480555534362793, -6.659245014190674, 0.00015050164074636996, 
    0.0006666610715910792, 22.50022315979004, -2.5853612422943115, 
    -6.1252665519714355, 21.745553970336914, 0.0, 0.0, 0.0, 0.0, 
    -0.0015395419904962182, 0.0015926380874589086
], dtype=np.float64)

M     = cv2.getPerspectiveTransform(SRC, DST)
M_inv = cv2.getPerspectiveTransform(DST, SRC)

# ── Lane geometry offsets (bird's-eye px) ─────────────────────────────────
L2C_OFFSET = 300    # yellow left  → center
C2R_OFFSET = 310    # center       → right virtual edge

# ── ROI (bird's-eye px) ───────────────────────────────────────────────────
ROI_TOP   = 220
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

# ── Debug tile size ───────────────────────────────────────────────────────
DEBUG_W = 480
DEBUG_H = 360


# ═══════════════════════════════════════════════════════════════════════════
#  YELLOW TAPE CONFIG  (v2 — PRIMARY, left edge)
#  Tune these without touching anything in the DASH block below.
# ═══════════════════════════════════════════════════════════════════════════

Y_HSV_LOW  = np.array([ 27, 100, 140], dtype=np.uint8)
Y_HSV_HIGH = np.array([ 35, 255, 255], dtype=np.uint8)

Y_OPEN_K   = (4,  4)   # morphological open  kernel — removes noise
Y_CLOSE_K  = (13, 13)  # morphological close kernel — fills tape gaps

Y_MIN_AREA = 400        # minimum contour area to be considered tape
Y_MIN_PTS  = 50         # minimum pixel count for a valid polyfit

Y_SEARCH_WINDOW          = 120   # px, normal tracking window around last x
Y_SEARCH_WINDOW_RECOVERY = 400   # px, wider window when recovering

Y_MAX_LAT_JUMP  = 80    # px   — reject fit if lateral jump exceeds this
Y_MAX_CURV_JUMP = 0.12  # coeff — reject fit if curvature jump exceeds this
Y_SMOOTH_N      = 15    # history depth for smoothing
Y_RECOVERY_THR  = 8     # consecutive rejects before entering recovery


# ═══════════════════════════════════════════════════════════════════════════
#  CENTER DASH CONFIG  (v1 — SECONDARY, white dashed line)
#  Tune these without touching anything in the YELLOW block above.
# ═══════════════════════════════════════════════════════════════════════════

D_CLAHE_CLIP  = 1.5
D_CLAHE_GRID  = (8, 8)
D_ADAPT_BLOCK = 199     # adaptive threshold block size (must be odd)
D_ADAPT_C     = -45     # adaptive threshold constant
D_CANNY_LOW   = 215
D_CANNY_HIGH  = 170

D_MIN_AREA     = 120    # contour area filter — min
D_MAX_AREA     = 2000   # contour area filter — max
D_MIN_ASPECT   = 1.4    # h/w ratio — dashes are taller than wide
D_MIN_SOLIDITY = 0.7    # area / convex-hull area

D_OFFSET_FALLBACK_THR = 4   # consecutive rejects before snapping to yellow+offset
                              # (must be <= D_RECOVERY_THR)
D_OUTPUT_SMOOTH_N   = 6     # frames to smooth the final center output
D_OUTPUT_MAX_LAT    = 60    # px — max jump allowed on the rendered center line
                             # applies to BOTH detect and offset candidates


D_SEARCH_WINDOW          = 100   # px, normal
D_SEARCH_WINDOW_RECOVERY = 400   # px, recovery

D_MAX_LAT_JUMP  = 50   # px
D_MAX_CURV_JUMP = 0.01
D_SMOOTH_N      = 15
D_RECOVERY_THR  = 8

# Pre-compute CLAHE for dash pipeline only
_DASH_CLAHE = cv2.createCLAHE(
    clipLimit=D_CLAHE_CLIP, tileGridSize=D_CLAHE_GRID)


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — YELLOW TAPE  (v2 logic, untouched)
# ═══════════════════════════════════════════════════════════════════════════

def yellow_extract_mask(bgr_warped: np.ndarray) -> np.ndarray:
    """HSV threshold → morphological clean → binary yellow mask."""
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
    """Keep only contours whose area >= Y_MIN_AREA."""
    out = np.zeros_like(mask)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) >= Y_MIN_AREA:
            cv2.drawContours(out, [cnt], -1, 255, -1)
    return out


def yellow_fit(filtered: np.ndarray,
               last_x,
               recovery: bool):
    """
    Histogram peak → search window → 2nd-degree polyfit on yellow pixels.

    Returns (coeffs | None, ys, xs, left_x_bottom)
    """
    hist   = np.sum(filtered, axis=0).astype(np.float32)
    window = (Y_SEARCH_WINDOW_RECOVERY if recovery else Y_SEARCH_WINDOW)

    if last_x is not None:
        lo = max(0, int(last_x) - window)
        hi = min(UNWARPED_SIZE[0], int(last_x) + window)
        w2 = np.zeros_like(hist)
        w2[lo:hi] = hist[lo:hi]
        if w2.max() >= 5:
            hist = w2

    if hist.max() < 5:
        return None, np.array([]), np.array([]), last_x

    peak  = int(np.argmax(hist))
    lo    = max(0, peak - window)
    hi    = min(UNWARPED_SIZE[0], peak + window)
    locked = np.zeros_like(filtered)
    locked[:, lo:hi] = filtered[:, lo:hi]

    ys, xs = np.where(locked > 0)
    if len(ys) < Y_MIN_PTS:
        return None, ys, xs, last_x

    coeffs = np.polyfit(ys, xs, 2)
    bot    = float(np.polyval(coeffs, UNWARPED_SIZE[1] - 1))
    return coeffs, ys, xs, bot


def yellow_is_valid(new_c: np.ndarray, history: deque):
    """
    Validate a new yellow fit against the smoothed history.
    Returns (valid, lat_jump, curv_jump).
    Uses Y_MAX_LAT_JUMP / Y_MAX_CURV_JUMP thresholds.
    """
    if len(history) == 0:
        return True, 0.0, 0.0
    last      = np.mean(history, axis=0)
    lat_jump  = abs(np.polyval(new_c, UNWARPED_SIZE[1] - 1)
                    - np.polyval(last, UNWARPED_SIZE[1] - 1))
    curv_jump = abs(new_c[0] - last[0])
    valid     = (lat_jump  <= Y_MAX_LAT_JUMP and
                 curv_jump <= Y_MAX_CURV_JUMP)
    return valid, float(lat_jump), float(curv_jump)


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — CENTER DASHES  (v1 logic, untouched)
# ═══════════════════════════════════════════════════════════════════════════

def dash_preprocess(gray: np.ndarray):
    """
    v1 pipeline: CLAHE → adaptive threshold → morphological open
                → GaussianBlur → Canny → AND with tape mask.
    Returns (enhanced, edges_gated, tape_mask).
    """
    enhanced  = _DASH_CLAHE.apply(gray)
    tape_mask = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        D_ADAPT_BLOCK, D_ADAPT_C
    )
    open_k    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_OPEN, open_k)
    blurred   = cv2.GaussianBlur(enhanced, (5, 5), 0)
    canny     = cv2.Canny(blurred, D_CANNY_LOW, D_CANNY_HIGH)
    edges_gated = cv2.bitwise_and(canny, tape_mask)
    return enhanced, edges_gated, tape_mask


def dash_filter_contours(roi_mask: np.ndarray) -> np.ndarray:
    """
    v1 contour filter: morphological close + dilate, then
    area / aspect / solidity checks.
    """
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


def dash_find_center_x(filtered: np.ndarray,
                        last_cx,
                        recovery: bool):
    """
    v1 histogram search: find the dominant column of dash pixels.
    Returns best center_x (int) or last_cx if nothing found.
    """
    hist   = np.sum(filtered, axis=0).astype(np.float32)
    window = (D_SEARCH_WINDOW_RECOVERY if recovery else D_SEARCH_WINDOW)

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


def dash_fit(filtered: np.ndarray,
             center_x,
             recovery: bool):
    """
    v1 window-lock → 2nd-degree polyfit on dash pixels.

    Returns (coeffs | None, ys, xs)
    """
    if center_x is None:
        ys, xs = np.where(filtered > 0)
        locked = filtered
    else:
        window = (D_SEARCH_WINDOW_RECOVERY if recovery else D_SEARCH_WINDOW)
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
    """
    Validate a new dash fit against the smoothed history.
    Returns (valid, lat_jump, curv_jump).
    Uses D_MAX_LAT_JUMP / D_MAX_CURV_JUMP thresholds.
    """
    if len(history) == 0:
        return True, 0.0, 0.0
    last      = np.mean(history, axis=0)
    lat_jump  = abs(np.polyval(new_c, UNWARPED_SIZE[1] - 1)
                    - np.polyval(last, UNWARPED_SIZE[1] - 1))
    curv_jump = abs(new_c[0] - last[0])
    valid     = (lat_jump  <= D_MAX_LAT_JUMP and
                 curv_jump <= D_MAX_CURV_JUMP)
    return valid, float(lat_jump), float(curv_jump)


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
    colour = np.zeros((OCC_ROWS, OCC_COLS, 3), dtype=np.uint8)
    colour[grid == OCC_FREE]     = (255, 255, 255)
    colour[grid == OCC_UNKNOWN]  = (30,  30,  30)
    colour[grid == OCC_OCCUPIED] = (0,   0,   0)

    disp = cv2.resize(colour, (OCC_DISPLAY_W, OCC_DISPLAY_H),
                      interpolation=cv2.INTER_NEAREST)

    cell_w, cell_h = OCC_DISPLAY_W / OCC_COLS, OCC_DISPLAY_H / OCC_ROWS
    gc = (60, 60, 60)
    for c in range(OCC_COLS + 1):
        cv2.line(disp, (int(c * cell_w), 0),
                 (int(c * cell_w), OCC_DISPLAY_H), gc, 1)
    for r in range(OCC_ROWS + 1):
        cv2.line(disp, (0, int(r * cell_h)),
                 (OCC_DISPLAY_W, int(r * cell_h)), gc, 1)

    legend = np.zeros((140, OCC_DISPLAY_W, 3), dtype=np.uint8)

    def swatch(x, y, col, lbl):
        cv2.rectangle(legend, (x, y), (x + 18, y + 14), col, -1)
        if col == (255, 255, 255):
            cv2.rectangle(legend, (x, y), (x + 18, y + 14), (100, 100, 100), 1)
        cv2.putText(legend, lbl, (x + 22, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

    swatch(10,  6, (255, 255, 255), 'Free / driveable')
    swatch(10, 24, (0,   0,   0),   'Boundary / occupied')
    swatch(10, 42, (30,  30,  30),  'Unknown / outside ROI')

    sc = ((0, 220, 0) if metrics['status'].startswith('OK')
          else (0, 100, 255))
    cv2.putText(legend, f"Status : {metrics['status']}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sc, 1)
    cv2.putText(legend,
                f"Yellow: {metrics['y_mode']}  "
                f"Dash: {metrics['d_mode']}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(legend,
                f"Yellow pts: {metrics['pts_yellow']:3d}   "
                f"Dash pts: {metrics['pts_dash']:3d}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(legend,
                f"Grid {OCC_COLS}x{OCC_ROWS}  "
                f"({UNWARPED_SIZE[0]/OCC_COLS:.1f}x"
                f"{UNWARPED_SIZE[1]/OCC_ROWS:.1f} px/cell)",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

    title = np.zeros((30, OCC_DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(title, 'OCCUPANCY GRID  |  v3 Yellow+Dash Pipeline',
                (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    return np.vstack([title, disp, legend])


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def add_label(img: np.ndarray, text: str) -> np.ndarray:
    out   = img.copy()
    bar_w = min(len(text) * 10 + 14, img.shape[1])
    cv2.rectangle(out, (0, 0), (bar_w, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    return out


def to_tile(img: np.ndarray, w: int = DEBUG_W, h: int = DEBUG_H) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (w, h))


def draw_metrics_panel(img: np.ndarray, m: dict) -> np.ndarray:
    panel   = img.copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (560, 380), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)

    C_OK = (0,255,0); C_WARN=(0,165,255); C_BAD=(0,0,255)
    C_INFO=(255,255,255); C_YEL=(0,220,220); C_DASH=(200,200,100)

    def put(t, y, c=C_INFO):
        cv2.putText(panel, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, c, 1)

    sc = (C_OK   if m['status'].startswith('OK')  else
          C_WARN if m['status'].startswith('REJ') else C_BAD)

    put(f"Status       : {m['status']}",  26, sc)
    put(f"Frame        : {m.get('frame', 0):6d}", 50, C_INFO)

    # ── Yellow block ──────────────────────────────────────────────────────
    put("── YELLOW (primary left edge) ──────────────", 78, C_YEL)
    put(f"  mode : {m['y_mode']}   pts : {m['pts_yellow']:3d}   "
        f"recovery : {'YES' if m['y_recovery'] else 'no'}",
        100,
        C_OK if m['pts_yellow'] >= Y_MIN_PTS else C_BAD)
    put(f"  A={m['y_A']:+.6f}  B={m['y_B']:+.4f}  "
        f"left_x_bot={m['y_bot']:.1f}px", 122, C_YEL)
    put(f"  lat_jump={m['y_lat']:.2f}px (max {Y_MAX_LAT_JUMP})  "
        f"curv_jump={m['y_curv']:.5f} (max {Y_MAX_CURV_JUMP})",
        144,
        C_OK if m['y_lat'] < Y_MAX_LAT_JUMP else C_BAD)

    # ── Dash block ────────────────────────────────────────────────────────
    put("── DASH  (secondary center) ────────────────", 172, C_DASH)
    put(f"  mode : {m['d_mode']}   pts : {m['pts_dash']:3d}   "
        f"recovery : {'YES' if m['d_recovery'] else 'no'}",
        194,
        C_OK if m['d_mode'] == 'DETECT' else C_WARN)
    put(f"  A={m['d_A']:+.6f}  B={m['d_B']:+.4f}", 216, C_DASH)
    put(f"  lat_jump={m['d_lat']:.2f}px (max {D_MAX_LAT_JUMP})  "
        f"curv_jump={m['d_curv']:.5f} (max {D_MAX_CURV_JUMP})",
        238,
        C_OK if m['d_lat'] < D_MAX_LAT_JUMP else C_BAD)

    # ── Right / offsets ───────────────────────────────────────────────────
    put("── RIGHT (offset from center) ──────────────", 266, C_INFO)
    put(f"  L2C={L2C_OFFSET}px   C2R={C2R_OFFSET}px", 288, C_INFO)

    # ── Tuning summary ────────────────────────────────────────────────────
    put(f"Y: HSV[{Y_HSV_LOW[0]}-{Y_HSV_HIGH[0]}] "
        f"smooth={Y_SMOOTH_N} rec_thr={Y_RECOVERY_THR}",
        316, C_YEL)
    put(f"D: adaptB={D_ADAPT_BLOCK} C={D_ADAPT_C} "
        f"smooth={D_SMOOTH_N} rec_thr={D_RECOVERY_THR}",
        338, C_DASH)
    put(f"D: area=[{D_MIN_AREA},{D_MAX_AREA}] "
        f"asp={D_MIN_ASPECT} sol={D_MIN_SOLIDITY}",
        360, C_DASH)

    return panel


def build_fit_vis(ys_y, xs_y, ys_d, xs_d,
                  left_x, center_x, right_x,
                  ploty, l_valid, c_valid, r_valid) -> np.ndarray:
    vis = np.zeros((UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)

    # Raw pixels
    if len(ys_y) > 0:
        ys_y_c = np.clip(ys_y, 0, UNWARPED_SIZE[1]-1)
        xs_y_c = np.clip(xs_y, 0, UNWARPED_SIZE[0]-1)
        vis[ys_y_c, xs_y_c] = (0, 215, 255)          # yellow tape pixels

    if len(ys_d) > 0:
        ys_d_c = np.clip(ys_d, 0, UNWARPED_SIZE[1]-1)
        xs_d_c = np.clip(xs_d, 0, UNWARPED_SIZE[0]-1)
        vis[ys_d_c, xs_d_c] = (200, 200, 200)         # dash pixels

    # Fitted curves
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


def draw_roi_on_warped(warped: np.ndarray) -> np.ndarray:
    vis = warped.copy()
    cv2.rectangle(vis,
                  (ROI_LEFT,  ROI_TOP),
                  (ROI_RIGHT, UNWARPED_SIZE[1] - 1),
                  (0, 255, 255), 2)
    cv2.putText(vis, 'ROI', (ROI_LEFT + 4, ROI_TOP + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return vis


def build_debug_mosaic(undist, warped,
                       yellow_mask, yellow_filt,
                       edges_gated, dash_filt,
                       fit_vis, overlay_frame,
                       occ_display) -> np.ndarray:
    """
    9 sources → 8-panel 4×2 mosaic
    ┌──────────┬──────────┬──────────┬──────────┐
    │1 Undist  │2 BirdEye │3 YelHSV  │4 YelFilt │
    ├──────────┼──────────┼──────────┼──────────┤
    │5 DashEdge│6 DashFilt│7 PolyFit │8 Overlay │
    └──────────┴──────────┴──────────┴──────────┘
    Occupancy is embedded inside panel 7 legend and shown separately.
    """
    warped_roi = draw_roi_on_warped(warped)

    t1 = add_label(to_tile(undist),        "1: Undistorted")
    t2 = add_label(to_tile(warped_roi),    "2: Bird's-Eye + ROI")
    t3 = add_label(to_tile(yellow_mask),   "3: Yellow HSV Mask")
    t4 = add_label(to_tile(yellow_filt),   "4: Yellow Filtered")
    t5 = add_label(to_tile(edges_gated),   "5: Dash Edges (v1)")
    t6 = add_label(to_tile(dash_filt),     "6: Dash Filtered (v1)")
    t7 = add_label(to_tile(fit_vis),       "7: Poly Fit Y/C/R")
    t8 = add_label(
        cv2.resize(occ_display, (DEBUG_W, DEBUG_H)),
        "8: Occupancy Grid"
    )

    row1 = np.hstack([t1, t2, t3, t4])
    row2 = np.hstack([t5, t6, t7, t8])
    return np.vstack([row1, row2])


# ═══════════════════════════════════════════════════════════════════════════
#  STATEFUL LANE DETECTOR  (two fully independent tracking pipelines)
# ═══════════════════════════════════════════════════════════════════════════

class LaneDetector:
    """
    Yellow left edge  → v2 HSV pipeline,  own history / thresholds
    Center dashes     → v1 adaptive pipeline, own history / thresholds
    Right edge        → always center + C2R_OFFSET
    """

    def __init__(self,
                 l2c: int = L2C_OFFSET,
                 c2r: int = C2R_OFFSET):
        self.l2c = l2c
        self.c2r = c2r

        # ── Yellow tracking state (v2) ────────────────────────────────────
        self._y_history   = deque(maxlen=Y_SMOOTH_N)
        self._y_last_x    = None    # last known left_x_bottom
        self._y_rejects   = 0
        self._y_recovery  = False

        # ── Dash tracking state (v1) ──────────────────────────────────────
        self._d_history   = deque(maxlen=D_SMOOTH_N)
        self._d_last_cx   = None    # last known histogram center_x
        self._d_rejects   = 0
        self._d_recovery  = False

        self._d_consec_rejects = 0   # counts every reject/low-pts frame in a row
        self._center_out_history = deque(maxlen=D_OUTPUT_SMOOTH_N)  # smoothed output

        self._frame_idx   = 0

        self.metrics = {
            'status':     'INIT',
            'frame':      0,
            # yellow
            'y_mode':     'INIT',  'pts_yellow': 0,
            'y_recovery': False,   'y_bot':      0.0,
            'y_A':        0.0,     'y_B':        0.0,
            'y_lat':      0.0,     'y_curv':     0.0,
            # dash
            'd_mode':     'INIT',  'pts_dash':   0,
            'd_recovery': False,
            'd_A':        0.0,     'd_B':        0.0,
            'd_lat':      0.0,     'd_curv':     0.0,
        }

    # ─────────────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray):
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
        yellow_roi[:ROI_TOP, :] = 0                  # apply ROI_TOP

        yellow_filt = yellow_filter_contours(yellow_roi)

        (y_coeffs_new,
         ys_y, xs_y,
         y_bot) = yellow_fit(yellow_filt,
                             self._y_last_x,
                             self._y_recovery)

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
                self._d_consec_rejects = 0   # ← reset consecutive counter
                self._d_recovery       = False
            else:
                self._d_rejects        += 1
                self._d_consec_rejects += 1
                if self._d_rejects >= D_RECOVERY_THR:
                    self._d_recovery = True
                    self._d_last_cx  = None
                    self._d_history.clear()  # ← flush stale history
                d_mode = 'REJ'
        else:
            self._d_rejects        += 1
            self._d_consec_rejects += 1
            if self._d_rejects >= D_RECOVERY_THR:
                self._d_recovery = True
                self._d_last_cx  = None
                self._d_history.clear()      # ← flush stale history
            d_mode = 'LOW_PTS'

        # Use smoothed dash history if available AND not lost too long
        if self._d_history and self._d_consec_rejects < D_OFFSET_FALLBACK_THR:
            center_coeffs = np.mean(self._d_history, axis=0)
        else:
            # Snap to yellow + offset — no stale dash poly will be drawn
            center_coeffs = None
            if d_mode == 'DETECT':
                d_mode = 'OFFSET'

        pts_dash = len(ys_d)

        # ── Center fallback: offset from yellow if dashes lost ────────────
        # ── Candidate center: detected dashes OR yellow+offset ────────────
        if center_coeffs is None and left_coeffs is not None:
            # Build offset candidate from current yellow poly
            candidate         = left_coeffs.copy()
            candidate[2]      = left_coeffs[2] + self.l2c
            d_mode            = 'OFFSET'
        elif center_coeffs is not None:
            candidate = center_coeffs   # from dash history
        else:
            candidate = None            # no yellow either — nothing to draw

        # ── Output smoother: validate candidate lateral jump ──────────────
        # This is the single gate that prevents snapping regardless of mode.
        if candidate is not None:
            if len(self._center_out_history) > 0:
                last_out  = np.mean(self._center_out_history, axis=0)
                jump      = abs(
                    np.polyval(candidate, UNWARPED_SIZE[1] - 1) -
                    np.polyval(last_out,  UNWARPED_SIZE[1] - 1)
                )
                if jump <= D_OUTPUT_MAX_LAT:
                    # Accepted — push into output history
                    self._center_out_history.append(candidate)
                else:
                    # Too large — hold last output, do NOT update history
                    # (history will naturally drift toward candidate over
                    #  D_OUTPUT_SMOOTH_N frames)
                    pass
            else:
                # First frame — accept unconditionally
                self._center_out_history.append(candidate)

        # Final center coeffs used for rendering and occupancy
        center_coeffs = (np.mean(self._center_out_history, axis=0)
                         if self._center_out_history else None)



        # ══════════════════════════════════════════════════════════════════
        #  RIGHT EDGE + SAMPLE CURVES
        # ══════════════════════════════════════════════════════════════════
        ploty = np.linspace(ROI_TOP, UNWARPED_SIZE[1] - 1, 200)

        left_x = (np.polyval(left_coeffs, ploty)
                  if left_coeffs   is not None
                  else np.full_like(ploty, 0.0))
        center_x = (np.polyval(center_coeffs, ploty)
                    if center_coeffs is not None
                    else left_x + self.l2c)
        right_x = center_x + self.c2r

        l_valid = (left_x   >= 0) & (left_x   < UNWARPED_SIZE[0])
        c_valid = (center_x >= 0) & (center_x < UNWARPED_SIZE[0])
        r_valid = (right_x  >= 0) & (right_x  < UNWARPED_SIZE[0])

        # ══════════════════════════════════════════════════════════════════
        #  METRICS
        # ══════════════════════════════════════════════════════════════════
        status = ('OK' if left_coeffs is not None else 'NO_YELLOW')

        self.metrics.update({
            'status':     status,
            'frame':      self._frame_idx,
            'y_mode':     y_mode,
            'pts_yellow': pts_yellow,
            'y_recovery': self._y_recovery,
            'y_bot':      float(np.polyval(left_coeffs, UNWARPED_SIZE[1]-1))
                          if left_coeffs is not None else 0.0,
            'y_A': float(left_coeffs[0])   if left_coeffs   is not None else 0.0,
            'y_B': float(left_coeffs[1])   if left_coeffs   is not None else 0.0,
            'y_lat':  y_lat,   'y_curv':  y_curv,
            'd_mode':     d_mode,
            'pts_dash':   pts_dash,
            'd_recovery': self._d_recovery,
            'd_A': float(center_coeffs[0]) if center_coeffs is not None else 0.0,
            'd_B': float(center_coeffs[1]) if center_coeffs is not None else 0.0,
            'd_lat':  d_lat,   'd_curv':  d_curv,
        })

        # ══════════════════════════════════════════════════════════════════
        #  OUTPUTS
        # ══════════════════════════════════════════════════════════════════
        occ_grid    = build_occupancy_grid(left_coeffs, right_x, ploty)
        occ_display = render_occupancy_grid(occ_grid, self.metrics)

        fit_vis = build_fit_vis(
            ys_y, xs_y, ys_d, xs_d,
            left_x, center_x, right_x,
            ploty, l_valid, c_valid, r_valid
        )

        # ── Polygon overlay → unwarp back ─────────────────────────────────
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
            cv2.polylines(bird_ov, [pts_c], False, (255, 255, 0), 3)
        if len(pts_r) > 1:
            cv2.polylines(bird_ov, [pts_r], False, (0, 0, 255), 3)

        overlay_inv   = cv2.warpPerspective(
            bird_ov, M_inv, ORIGINAL_SIZE, flags=cv2.INTER_LINEAR)
        overlay_frame = cv2.addWeighted(undist, 1.0, overlay_inv, 0.5, 0)
        overlay_frame = draw_metrics_panel(overlay_frame, self.metrics)

        debug_mosaic = build_debug_mosaic(
            undist, warped,
            yellow_mask, yellow_filt,
            edges_gated, dash_filt,
            fit_vis, overlay_frame,
            occ_display
        )

        return overlay_frame, debug_mosaic, occ_display, occ_grid


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO I/O
# ═══════════════════════════════════════════════════════════════════════════

def open_writer(path, w, h, fps):
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer


def progress_bar(cur, total, bar_len=50):
    filled = int(bar_len * cur / max(total, 1))
    bar    = '█' * filled + '░' * (bar_len - filled)
    return f"\r[{bar}] {cur:6d}/{total:6d}  ({100*cur/max(total,1):5.1f}%)"


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Lane Detection v3 — video inference')
    p.add_argument('--input',  '-i', required=True)
    p.add_argument('--out-debug',   default='debug_grid.mp4')
    p.add_argument('--out-overlay', default='lane_overlay.mp4')
    p.add_argument('--out-occ',     default='occupancy_grid.mp4')
    p.add_argument('--no-display',  action='store_true')
    p.add_argument('--l2c-offset',  type=int, default=L2C_OFFSET)
    p.add_argument('--c2r-offset',  type=int, default=C2R_OFFSET)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--end-frame',   type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Not found: {args.input}"); sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.input}"); sys.exit(1)

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'─'*62}")
    print(f"  Input  : {args.input}  [{src_w}x{src_h}  {fps:.2f}fps  {total}fr]")
    print(f"{'─'*62}\n")

    start = max(0, args.start_frame)
    end   = total if args.end_frame < 0 else min(args.end_frame, total)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    n_frames = end - start

    detector = LaneDetector(l2c=args.l2c_offset, c2r=args.c2r_offset)

    # ── First frame — determine output sizes ──────────────────────────────
    ret, first = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame."); sys.exit(1)

    ov_f, dbg_f, occ_f, _ = detector.process_frame(first.copy())
    ov_h,  ov_w  = ov_f.shape[:2]
    dbg_h, dbg_w = dbg_f.shape[:2]
    occ_h, occ_w = occ_f.shape[:2]

    try:
        w_ov  = open_writer(args.out_overlay, ov_w,  ov_h,  fps)
        w_dbg = open_writer(args.out_debug,   dbg_w, dbg_h, fps)
        w_occ = open_writer(args.out_occ,     occ_w, occ_h, fps)
    except RuntimeError as e:
        print(f"[ERROR] {e}"); cap.release(); sys.exit(1)

    print(f"  overlay  → {args.out_overlay}  [{ov_w}x{ov_h}]")
    print(f"  debug    → {args.out_debug}    [{dbg_w}x{dbg_h}]")
    print(f"  occupancy→ {args.out_occ}      [{occ_w}x{occ_h}]\n")

    w_ov.write(ov_f); w_dbg.write(dbg_f); w_occ.write(occ_f)

    SHOW = not args.no_display
    if SHOW:
        PREV_W = 1280
        PREV_H = int(dbg_h * PREV_W / max(dbg_w, 1))
        cv2.namedWindow('Debug',     cv2.WINDOW_NORMAL)
        cv2.namedWindow('Overlay',   cv2.WINDOW_NORMAL)
        cv2.namedWindow('Occupancy', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug',    PREV_W, PREV_H)
        cv2.resizeWindow('Overlay',  1280, 720)
        cv2.resizeWindow('Occupancy', occ_w, occ_h)
        cv2.imshow('Debug',     dbg_f)
        cv2.imshow('Overlay',   cv2.resize(ov_f, (1280, 720)))
        cv2.imshow('Occupancy', occ_f)
        cv2.waitKey(1)

    for fn in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        ov_f, dbg_f, occ_f, _ = detector.process_frame(frame)
        w_ov.write(ov_f); w_dbg.write(dbg_f); w_occ.write(occ_f)

        print(progress_bar(fn, n_frames), end='', flush=True)

        if SHOW:
            cv2.imshow('Debug',     dbg_f)
            cv2.imshow('Overlay',   cv2.resize(ov_f, (1280, 720)))
            cv2.imshow('Occupancy', occ_f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Quit."); break

    print(f"\n\n{'─'*62}")
    print(f"  Done.  {n_frames} frames")
    print(f"  {args.out_overlay}")
    print(f"  {args.out_debug}")
    print(f"  {args.out_occ}")
    print(f"{'─'*62}\n")

    cap.release()
    w_ov.release(); w_dbg.release(); w_occ.release()

    if SHOW:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()