#!/usr/bin/env python3
"""
Standalone Lane Detection CV Pipeline
======================================
File   : lanev2_vid_inference.py
Author : Sumit
System : Ubuntu 22.04 | OpenCV 4.x | NumPy

Takes a recorded video and exports 3 output videos:
    1. debug_grid.mp4       — 8-panel debug mosaic
    2. lane_overlay.mp4     — Polygon overlay on road video
    3. occupancy_grid.mp4   — Occupancy grid only

Lane Detection Strategy:
    PRIMARY   : Yellow tape on left edge  (HSV color mask)
    SECONDARY : Center dashed lines       (adaptive threshold + dash filter)
                When center dashes missing → estimated from yellow + offset
    RIGHT     : Always offset-based       (no CV tracking)

Usage:
    python3 lanev2_vid_inference.py --input cpm_oakd.mp4
    python3 lanev2_vid_inference.py --input cpm_oakd.mp4 --no-display
    python3 lanev2_vid_inference.py --input cpm_oakd.mp4 \
                                  --out-debug  my_debug.mp4 \
                                  --out-overlay my_overlay.mp4 \
                                  --out-occ    my_occ.mp4
"""

import argparse
import sys
import os
import cv2
import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS / CONFIG
# ═══════════════════════════════════════════════════════════════════════════

UNWARPED_SIZE = (500, 600)      # (width, height) bird's-eye canvas
ORIGINAL_SIZE = (1920, 1080)    # expected input resolution

# ── Perspective warp control points ──────────────────────────────────────
# SRC: four corners in the original (undistorted) image
# DST: corresponding corners in bird's-eye canvas

SRC = np.array([
    (675, 543),  # TL
    (1247, 517),  # TR
    (1883, 851),  # BR
    (43, 882)   # BL
], dtype=np.float32)

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
YELLOW_HSV_LOW  = np.array([ 27,  100,  140], dtype=np.uint8)
YELLOW_HSV_HIGH = np.array([ 35, 255, 255], dtype=np.uint8)

YELLOW_OPEN_K  = (4,  4)
YELLOW_CLOSE_K = (13, 13)
YELLOW_MIN_AREA = 400
YELLOW_MIN_PTS  = 50

# ══════════════════════════════════════════════════════════════════════════
#  CENTER DASHED LINE CONFIG  (SECONDARY)
# ══════════════════════════════════════════════════════════════════════════
CANNY_LOW   = 255
CANNY_HIGH  = 255
CLAHE_CLIP  = 1.5
CLAHE_GRID  = (8, 8)
ADAPT_BLOCK = 199
ADAPT_C     = -25

MIN_DASH_AREA     = 120
MAX_DASH_AREA     = 1800
MIN_DASH_ASPECT   = 1.4
MIN_DASH_SOLIDITY = 0.7
CENTER_MIN_PTS    = 40

# ══════════════════════════════════════════════════════════════════════════
#  LANE GEOMETRY OFFSETS  (bird's-eye pixel space)
# ══════════════════════════════════════════════════════════════════════════
LEFT_TO_CENTER_OFFSET  = 270   # yellow_left → center
CENTER_TO_RIGHT_OFFSET = 290   # center      → right virtual edge

# ══════════════════════════════════════════════════════════════════════════
#  ROI  (bird's-eye pixel space)
# ══════════════════════════════════════════════════════════════════════════
ROI_TOP   = 200
ROI_LEFT  = 0
ROI_RIGHT = 485

# ══════════════════════════════════════════════════════════════════════════
#  TRACKING / SMOOTHING
# ══════════════════════════════════════════════════════════════════════════
MAX_LATERAL_JUMP           = 80
MAX_CURV_JUMP              = 0.12
SMOOTH_N                   = 15
RECOVERY_THRESHOLD         = 8
YELLOW_SEARCH_WINDOW       = 120
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
        center = int(last_left_x)

        lo = max(0, center - window)
        hi = min(UNWARPED_SIZE[0], center + window)

        windowed = np.zeros_like(hist)
        windowed[lo:hi] = hist[lo:hi]

        if windowed.max() >= 5:
            hist = windowed

    if hist.max() < 5:
        return None, np.array([]), np.array([]), last_left_x

    peak_x = int(np.argmax(hist))
    lo     = int(max(0, peak_x - window))
    hi     = int(min(UNWARPED_SIZE[0], peak_x + window))
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
    close_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed   = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, close_k)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated  = cv2.dilate(closed, dilate_k, iterations=3)
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

        # Paint left boundary
        if 0.0 <= lx < UNWARPED_SIZE[0]:
            lx_c = int(lx / px_per_col)
            for dc in range(-1, 2):
                c = lx_c + dc
                if 0 <= c < OCC_COLS:
                    grid[row_idx, c] = OCC_OCCUPIED

        # Paint right boundary
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

    # Grid lines
    cell_w     = OCC_DISPLAY_W / OCC_COLS
    cell_h     = OCC_DISPLAY_H / OCC_ROWS
    grid_color = (60, 60, 60)
    for c in range(OCC_COLS + 1):
        x = int(c * cell_w)
        cv2.line(disp, (x, 0), (x, OCC_DISPLAY_H), grid_color, 1)
    for r in range(OCC_ROWS + 1):
        y = int(r * cell_h)
        cv2.line(disp, (0, y), (OCC_DISPLAY_W, y), grid_color, 1)

    # ── Legend panel ──────────────────────────────────────────────────────
    legend = np.zeros((140, OCC_DISPLAY_W, 3), dtype=np.uint8)

    def swatch(x: int, y: int, color: tuple, label: str):
        cv2.rectangle(legend, (x, y), (x + 18, y + 14), color, -1)
        if color == (255, 255, 255):
            cv2.rectangle(legend, (x, y), (x + 18, y + 14), (100, 100, 100), 1)
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
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(legend,
                f"Yellow pts: {metrics['pts_yellow']:3d}   "
                f"Dash pts: {metrics['pts_center']:3d}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(legend,
                f"Grid {OCC_COLS}x{OCC_ROWS}  "
                f"({UNWARPED_SIZE[0]/OCC_COLS:.1f}x"
                f"{UNWARPED_SIZE[1]/OCC_ROWS:.1f} px/cell)",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

    title = np.zeros((30, OCC_DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(title, 'OCCUPANCY GRID  |  Yellow-Primary Pipeline',
                (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    return np.vstack([title, disp, legend])


# ═══════════════════════════════════════════════════════════════════════════
#  DEBUG / VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a label bar into the top-left of the image."""
    out = img.copy()
    bar_w = min(len(text) * 10 + 14, img.shape[1])
    cv2.rectangle(out, (0, 0), (bar_w, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    return out


def to_tile(img: np.ndarray,
            w: int = DEBUG_W,
            h: int = DEBUG_H) -> np.ndarray:
    """Resize to tile dimensions; convert grayscale to BGR if needed."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (w, h))


def draw_roi_on_warped(warped: np.ndarray) -> np.ndarray:
    """
    Draw the active ROI rectangle and the warp source quad
    on the bird's-eye image for spatial context.
    """
    vis = warped.copy()
    # ROI box in bird's-eye space
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
    cv2.rectangle(overlay, (0, 0), (520, 340), (0, 0, 0), -1)
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

    put(f"[YELLOW] pts: {m['pts_yellow']:3d}  mode: {m['center_mode_left']}",
        76,
        C_OK if m['pts_yellow'] >= YELLOW_MIN_PTS else C_BAD)
    put(f"[YELLOW] A={m['A_left']:+.6f}  B={m['B_left']:+.4f}", 100, C_YEL)
    put(f"[YELLOW] left_x_bot = {m['left_x_bot']:.1f} px", 124, C_YEL)

    put(f"[CENTER] pts: {m['pts_center']:3d}  mode: {m['center_mode']}",
        148,
        C_OK   if m['center_mode'] == 'DETECT' else C_WARN)
    put(f"[CENTER] A={m['A_center']:+.6f}  B={m['B_center']:+.4f}", 172, C_INFO)

    put(f"[RIGHT]  always offset ({CENTER_TO_RIGHT_OFFSET}px from center)",
        196, C_INFO)

    put(f"Lat jump  : {m['lateral_jump']:6.2f} px  (max {MAX_LATERAL_JUMP})",
        220,
        C_OK if m['lateral_jump'] < MAX_LATERAL_JUMP else C_BAD)
    put(f"Curv jump : {m['curv_jump']:.5f}  (max {MAX_CURV_JUMP})",
        244,
        C_OK if m['curv_jump'] < MAX_CURV_JUMP else C_BAD)

    put(f"CLAHE={CLAHE_CLIP}  Canny={CANNY_LOW}/{CANNY_HIGH}  "
        f"AdaptB={ADAPT_BLOCK}", 268, C_INFO)
    put(f"L→C offset={LEFT_TO_CENTER_OFFSET}px  "
        f"C→R offset={CENTER_TO_RIGHT_OFFSET}px", 292, C_INFO)
    put(f"Frame: {m.get('frame', 0):6d}", 316, C_INFO)

    return panel


def build_poly_fit_vis(warped:      np.ndarray,
                       ys_y:        np.ndarray,
                       xs_y:        np.ndarray,
                       ys_d:        np.ndarray,
                       xs_d:        np.ndarray,
                       left_x:      np.ndarray,
                       center_x:    np.ndarray,
                       right_x:     np.ndarray,
                       ploty:       np.ndarray,
                       l_valid:     np.ndarray,
                       c_valid:     np.ndarray,
                       r_valid:     np.ndarray) -> np.ndarray:
    """
    Black-canvas visualisation showing:
      • Raw yellow pixels  (golden)
      • Raw dash pixels    (light grey)
      • Fitted left poly   (yellow dots)
      • Fitted center poly (cyan dots)
      • Fitted right poly  (red dots)

    Also draws the sliding-window search band for the yellow tape.
    """
    fit_vis = np.zeros(
        (UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)

    # Raw pixels
    if len(ys_y) > 0:
        fit_vis[ys_y, xs_y] = (0, 215, 255)
    if len(ys_d) > 0:
        fit_vis[ys_d, xs_d] = (200, 200, 200)

    # Polynomial curves
    for i in range(len(ploty)):
        cy = int(ploty[i])
        if l_valid[i]:
            cv2.circle(fit_vis, (int(left_x[i]),   cy), 2, (0, 215, 255), -1)
        if c_valid[i]:
            cv2.circle(fit_vis, (int(center_x[i]), cy), 2, (255, 255, 0), -1)
        if r_valid[i]:
            cv2.circle(fit_vis, (int(right_x[i]),  cy), 2, (0, 0, 255),   -1)

    # Legend text
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

    ┌────────────┬────────────┬────────────┬────────────┐
    │ 1 Undist   │ 2 Bird-Eye │ 3 YellHSV  │ 4 YelFilt  │
    ├────────────┼────────────┼────────────┼────────────┤
    │ 5 DashFilt │ 6 PolyFit  │ 7 Overlay  │ 8 Occupancy│
    └────────────┴────────────┴────────────┴────────────┘
    """
    # Draw ROI on the bird's-eye tile
    warped_roi = draw_roi_on_warped(warped)

    t1 = add_label(to_tile(undist,        DEBUG_W, DEBUG_H), "1: Undistorted")
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
    Wraps the full per-frame detection pipeline including tracking state.
    Call process_frame(bgr) → (result, debug_mosaic, occ_display, occ_grid)
    """

    def __init__(self,
                 l2c_offset: int = LEFT_TO_CENTER_OFFSET,
                 c2r_offset: int = CENTER_TO_RIGHT_OFFSET):

        self.l2c_offset = l2c_offset
        self.c2r_offset = c2r_offset

        # Yellow left edge state
        self._left_history  = deque(maxlen=SMOOTH_N)
        self._last_left_x   = None
        self._left_rejects  = 0
        self._left_recovery = False

        # Center dash state
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
        }

    # ─────────────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray):
        """
        Full per-frame detection.

        Returns
        -------
        overlay_frame : annotated original view (BGR)
        debug_mosaic  : 8-panel debug grid     (BGR)
        occ_display   : occupancy render        (BGR)
        occ_grid      : uint8 occupancy array   (OCC_ROWS x OCC_COLS)
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

        lat_jump = curv_jump = 0.0
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
            new_center = np.polyfit(ys_d, xs_d, 2)
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
            center_coeffs         = left_coeffs.copy()
            center_coeffs[2]      = left_coeffs[2] + self.l2c_offset

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
            warped, ys_y, xs_y, ys_d, xs_d,
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

        # Green fill between left and right
        if len(pts_l) > 2 and len(pts_r) > 2:
            fill = np.concatenate([pts_l, np.flipud(pts_r)], axis=0)
            cv2.fillPoly(overlay_bird, [fill], (0, 180, 0))

        # Left edge — yellow line
        if len(pts_l) > 1:
            cv2.polylines(overlay_bird, [pts_l], False, (0, 215, 255), 5)

        # Center — cyan line
        if len(pts_c) > 1:
            cv2.polylines(overlay_bird, [pts_c], False, (255, 255, 0), 3)

        # Right edge — red line
        if len(pts_r) > 1:
            cv2.polylines(overlay_bird, [pts_r], False, (0, 0, 255), 3)

        # Unwarp back to original perspective
        overlay_inv   = cv2.warpPerspective(
            overlay_bird, M_inv, ORIGINAL_SIZE, flags=cv2.INTER_LINEAR)
        overlay_frame = cv2.addWeighted(undist, 1.0, overlay_inv, 0.5, 0)
        overlay_frame = draw_metrics_panel(overlay_frame, self.metrics)

        # ── 12. Debug mosaic ──────────────────────────────────────────────
        debug_mosaic = build_debug_mosaic(
            undist,
            warped,
            yellow_mask,
            yellow_filtered,
            dash_filtered,
            fit_vis,
            overlay_frame,
            occ_display
        )

        return overlay_frame, debug_mosaic, occ_display, occ_grid


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO I/O HELPERS
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


def progress_bar(current: int,
                 total: int,
                 bar_len: int = 50) -> str:
    filled = int(bar_len * current / max(total, 1))
    bar    = '█' * filled + '░' * (bar_len - filled)
    pct    = 100.0 * current / max(total, 1)
    return f"\r[{bar}] {current:6d}/{total:6d}  ({pct:5.1f}%)"


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Standalone lane detection CV pipeline (video → 3 exports)')
    p.add_argument('--input',  '-i', required=True,
                   help='Input video file path')
    p.add_argument('--out-debug',   default='debug_grid.mp4',
                   help='Output debug mosaic video  (default: debug_grid.mp4)')
    p.add_argument('--out-overlay', default='lane_overlay.mp4',
                   help='Output overlay video        (default: lane_overlay.mp4)')
    p.add_argument('--out-occ',     default='occupancy_grid.mp4',
                   help='Output occupancy grid video (default: occupancy_grid.mp4)')
    p.add_argument('--no-display',  action='store_true',
                   help='Suppress live preview windows')
    p.add_argument('--l2c-offset',  type=int, default=LEFT_TO_CENTER_OFFSET,
                   help=f'Left → Centre offset px (default {LEFT_TO_CENTER_OFFSET})')
    p.add_argument('--c2r-offset',  type=int, default=CENTER_TO_RIGHT_OFFSET,
                   help=f'Centre → Right offset px (default {CENTER_TO_RIGHT_OFFSET})')
    p.add_argument('--start-frame', type=int, default=0,
                   help='First frame to process    (default 0)')
    p.add_argument('--end-frame',   type=int, default=-1,
                   help='Last frame to process     (-1 = all, default -1)')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Open input ────────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.input}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'─'*60}")
    print(f"  Input  : {args.input}")
    print(f"  Size   : {src_w}x{src_h}  →  pipeline: {ORIGINAL_SIZE[0]}x{ORIGINAL_SIZE[1]}")
    print(f"  FPS    : {fps:.2f}")
    print(f"  Frames : {total_frames}")
    print(f"{'─'*60}\n")

    # ── Seek to start frame ───────────────────────────────────────────────
    start_frame = max(0, args.start_frame)
    end_frame   = total_frames if args.end_frame < 0 else args.end_frame
    end_frame   = min(end_frame, total_frames)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames_to_process = end_frame - start_frame
    print(f"  Processing frames {start_frame} → {end_frame}  "
          f"({frames_to_process} frames)\n")

    # ── Initialise detector ───────────────────────────────────────────────
    detector = LaneDetector(
        l2c_offset=args.l2c_offset,
        c2r_offset=args.c2r_offset
    )

    # ── Process first frame to determine output sizes ─────────────────────
    ret, first = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame.")
        cap.release()
        sys.exit(1)

    overlay_f, debug_f, occ_f, _ = detector.process_frame(first.copy())

    ov_h,  ov_w  = overlay_f.shape[:2]   # ORIGINAL_SIZE
    dbg_h, dbg_w = debug_f.shape[:2]     # 4*DEBUG_W  x  2*DEBUG_H
    occ_h, occ_w = occ_f.shape[:2]       # OCC_DISPLAY_W x (title+OCC_DISPLAY_H+legend)

    # ── Open writers ──────────────────────────────────────────────────────
    try:
        w_debug   = open_writer(args.out_debug,   dbg_w, dbg_h, fps)
        w_overlay = open_writer(args.out_overlay, ov_w,  ov_h,  fps)
        w_occ     = open_writer(args.out_occ,     occ_w, occ_h, fps)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        cap.release()
        sys.exit(1)

    print(f"  Output 1 (debug)   : {args.out_debug}   [{dbg_w}x{dbg_h}]")
    print(f"  Output 2 (overlay) : {args.out_overlay} [{ov_w}x{ov_h}]")
    print(f"  Output 3 (occ)     : {args.out_occ}     [{occ_w}x{occ_h}]")
    print()

    # Write first frame
    w_debug.write(debug_f)
    w_overlay.write(overlay_f)
    w_occ.write(occ_f)

    # ── Preview windows ───────────────────────────────────────────────────
    SHOW = not args.no_display
    PREV_W = 1280
    PREV_H = int(dbg_h * PREV_W / max(dbg_w, 1))

    if SHOW:
        cv2.namedWindow('Debug Grid',      cv2.WINDOW_NORMAL)
        cv2.namedWindow('Lane Overlay',    cv2.WINDOW_NORMAL)
        cv2.namedWindow('Occupancy Grid',  cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug Grid',     PREV_W, PREV_H)
        cv2.resizeWindow('Lane Overlay',   1280,   720)
        cv2.resizeWindow('Occupancy Grid', OCC_DISPLAY_W, occ_h)

        cv2.imshow('Debug Grid',     debug_f)
        cv2.imshow('Lane Overlay',   cv2.resize(overlay_f, (1280, 720)))
        cv2.imshow('Occupancy Grid', occ_f)
        cv2.waitKey(1)

    # ── Main loop ─────────────────────────────────────────────────────────
    frame_num = 1   # first frame already processed

    while frame_num < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        overlay_f, debug_f, occ_f, _ = detector.process_frame(frame)

        w_debug.write(debug_f)
        w_overlay.write(overlay_f)
        w_occ.write(occ_f)

        frame_num += 1

        # Progress
        print(progress_bar(frame_num, frames_to_process), end='', flush=True)

        if SHOW:
            cv2.imshow('Debug Grid',     debug_f)
            cv2.imshow('Lane Overlay',   cv2.resize(overlay_f, (1280, 720)))
            cv2.imshow('Occupancy Grid', occ_f)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit key pressed — stopping early.")
                break

    # ── Cleanup ───────────────────────────────────────────────────────────
    print(f"\n\n{'─'*60}")
    print(f"  Processed  : {frame_num} frames")
    print(f"  Saved to:")
    print(f"    {args.out_debug}")
    print(f"    {args.out_overlay}")
    print(f"    {args.out_occ}")
    print(f"{'─'*60}\n")

    cap.release()
    w_debug.release()
    w_overlay.release()
    w_occ.release()

    if SHOW:
        print("  [Press any key in a preview window to close.]")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()