#!/usr/bin/env python3
"""
Lane Detection v5 — Standalone Video Inference
===============================================
File   : lanev5_vidout.py
Author : Sumit
System : Ubuntu 22.04 | OpenCV 4.x | NumPy

Yellow tape  (LEFT)   : v2 HSV logic  — fully independent config + helpers
Center dashes (CENTER): v1 adaptive-threshold logic — fully independent config + helpers
Right edge            : always CENTER + C2R_OFFSET

Exports 2 MP4s:
    1. debug_grid.mp4
    2. lane_overlay.mp4

Usage:
    python3 lanev5_vidout.py --input outer_run.mp4
    python3 lanev5_vidout.py --input road.mp4 --no-display
"""

import argparse
import csv
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

# ── Camera / vehicle reference (bird's-eye px) ────────────────────────────
CAMERA_CENTER_X = UNWARPED_SIZE[0] / 2.0   # = 250.0

# ── ROI (bird's-eye px) ───────────────────────────────────────────────────
ROI_TOP   = 220
ROI_LEFT  = 0
ROI_RIGHT = 485


# ══════════════════════════════════════════════════════════════════════════
#  DEBUG MOSAIC LAYOUT  (1920 × 1080 canvas)
# ══════════════════════════════════════════════════════════════════════════

CANVAS_W    = 1920
CANVAS_H    = 1080
PAD_OUTER   = 16
PAD_INNER   = 10
HEADER_H    = 40

_AVAIL_W    = CANVAS_W - 2 * PAD_OUTER
_AVAIL_H    = CANVAS_H - 2 * PAD_OUTER - HEADER_H

TOP_N       = 4
TOP_TILE_W  = (_AVAIL_W - (TOP_N - 1) * PAD_INNER) // TOP_N
TOP_TILE_H  = (_AVAIL_H - PAD_INNER) * 11 // 20

BOT_N       = 2
BOT_TILE_W  = (_AVAIL_W - (BOT_N - 1) * PAD_INNER) // BOT_N
BOT_TILE_H  = _AVAIL_H - TOP_TILE_H - PAD_INNER

TOP_ROW_Y   = PAD_OUTER + HEADER_H
BOT_ROW_Y   = TOP_ROW_Y + TOP_TILE_H + PAD_INNER

TOP_COL_X   = [PAD_OUTER + i * (TOP_TILE_W + PAD_INNER) for i in range(TOP_N)]
BOT_COL_X   = [PAD_OUTER + i * (BOT_TILE_W + PAD_INNER) for i in range(BOT_N)]

BG_COLOR     = (18,  18,  18)
HEADER_COLOR = (30,  30,  30)
BORDER_COLOR = (55,  55,  55)
LABEL_BG     = (0,   0,   0)
LABEL_FG     = (0,  220, 220)


# ═══════════════════════════════════════════════════════════════════════════
#  DEBUG MOSAIC HELPERS
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
    Resize *img* to (tw × th), add a label bar and an optional
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


def build_debug_mosaic_1080p(undist:        np.ndarray,
                              warped:        np.ndarray,
                              yellow_filt:   np.ndarray,
                              dash_filt:     np.ndarray,
                              fit_vis:       np.ndarray,
                              overlay_frame: np.ndarray,
                              metrics:       dict) -> np.ndarray:
    """
    Compose a 1920 × 1080 debug mosaic from 6 panels:

      Top row  (4 tiles): undistorted | bird's-eye+ROI |
                          yellow-filtered | dash-filtered
      Bottom row (2 tiles): poly-fit vis | lane overlay
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
    err_str = (f"{metrics.get('pixel_error', 0.0):+.1f}px"
               if metrics.get('llc_ok', False) else "n/a")
    cv2.putText(
        canvas,
        (f"sumit_laneCV v3  |  "
         f"Status: {metrics['status']}  |  "
         f"Frame: {metrics.get('frame', 0):6d}  |  "
         f"Y:{metrics['y_mode']} ({metrics['pts_yellow']}pts)  |  "
         f"D:{metrics['d_mode']} ({metrics['pts_dash']}pts)  |  "
         f"Y_rec:{'Y' if metrics['y_recovery'] else 'n'}  "
         f"D_rec:{'Y' if metrics['d_recovery'] else 'n'}  |  "
         f"err:{err_str}"),
        (PAD_OUTER + 32, PAD_OUTER + HEADER_H - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1,
        cv2.LINE_AA)

    tw, th = TOP_TILE_W, TOP_TILE_H
    bw, bh = BOT_TILE_W, BOT_TILE_H

    # ── top row ───────────────────────────────────────────────────────────

    # panel 1 — undistorted frame
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

    # panel 3 — yellow tape mask (golden false-colour)
    yf = np.zeros((*yellow_filt.shape, 3), dtype=np.uint8)
    yf[yellow_filt > 0] = (0, 215, 255)
    _paste(canvas,
           _make_tile(yf, tw, th, "3: Yellow Tape Filtered"),
           TOP_COL_X[2], TOP_ROW_Y)

    # panel 4 — dash mask (white false-colour)
    df = np.zeros((*dash_filt.shape, 3), dtype=np.uint8)
    df[dash_filt > 0] = (200, 200, 200)
    _paste(canvas,
           _make_tile(df, tw, th, "4: Dash Lines Filtered"),
           TOP_COL_X[3], TOP_ROW_Y)

    # ── bottom row ────────────────────────────────────────────────────────

    # panel 5 — polynomial fit visualisation
    _paste(canvas,
           _make_tile(fit_vis, bw, bh,
                      "5: Poly Fit  Yellow=left  Cyan=center  Red=right"),
           BOT_COL_X[0], BOT_ROW_Y)

    # panel 6 — final lane overlay
    _paste(canvas,
           _make_tile(overlay_frame, bw, bh, "6: Lane Overlay", metrics),
           BOT_COL_X[1], BOT_ROW_Y)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════
#  YELLOW TAPE CONFIG  (v2 — PRIMARY, left edge)
# ═══════════════════════════════════════════════════════════════════════════

Y_HSV_LOW  = np.array([ 20,  50, 130], dtype=np.uint8)
Y_HSV_HIGH = np.array([ 33, 255, 255], dtype=np.uint8)

Y_OPEN_K   = (4,  4)
Y_CLOSE_K  = (13, 13)

Y_MIN_AREA = 400
Y_MIN_PTS  = 50

Y_SEARCH_WINDOW          = 120
Y_SEARCH_WINDOW_RECOVERY = 400

Y_MAX_LAT_JUMP  = 80
Y_MAX_CURV_JUMP = 0.12
Y_SMOOTH_N      = 8
Y_RECOVERY_THR  = 8


# ═══════════════════════════════════════════════════════════════════════════
#  CENTER DASH CONFIG  (v1 — SECONDARY, white dashed line)
# ═══════════════════════════════════════════════════════════════════════════

D_CLAHE_CLIP  = 1.5
D_CLAHE_GRID  = (8, 8)
D_ADAPT_BLOCK = 199
D_ADAPT_C     = -45
D_CANNY_LOW   = 215
D_CANNY_HIGH  = 170

D_MIN_AREA       = 300   # raised from 120; eliminates small wall/edge artefacts
                         # (FP blobs measured at 103–218px; real dashes ≥ 374px)
D_MAX_AREA       = 2000
D_MIN_ELONGATION = 1.5
D_MIN_SOLIDITY   = 0.55

# Minimum TOTAL white-pixel count in dash_filt before attempting a poly fit.
# A single marginal blob (area ≈ 300–450) produces ~700–900 dilated pixels;
# two real dashes produce ≥ 1 200.  Setting the floor at 800 kills all
# single-small-blob false positives while accepting every genuine multi-dash
# frame (min observed: 778 px for a single large real dash near the ROI edge).
D_MIN_TOTAL_PTS  = 800

# Spatial-plausibility guard (bird's-eye px).
# The center dashed line must lie at least this far to the RIGHT of the yellow
# tape's bottom-row x.  A fit that falls to the left of (yellow + guard) is
# geometrically impossible (center line can never be left of the left boundary)
# and is discarded as a false positive.
# Guard is intentionally small (30 px) to be forgiving during tight turns
# where the two lines appear close together in bird's-eye view.
D_SPATIAL_GUARD  = 30

D_OFFSET_FALLBACK_THR = 8
D_OUTPUT_SMOOTH_N     = 6
D_OUTPUT_MAX_LAT      = 60

D_SEARCH_WINDOW          = 100
D_SEARCH_WINDOW_RECOVERY = 400
D_SEARCH_WINDOW_SEED     = 80

D_MAX_LAT_JUMP  = 50
D_MAX_CURV_JUMP = 0.01
D_SMOOTH_N      = 5
D_RECOVERY_THR  = 8

_DASH_CLAHE = cv2.createCLAHE(
    clipLimit=D_CLAHE_CLIP, tileGridSize=D_CLAHE_GRID)


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — YELLOW TAPE  (v2 logic)
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
    Returns (coeffs | None, ys, xs, left_x_bottom).
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
#  CV HELPERS — CENTER DASHES  (v1 logic)
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
        D_ADAPT_BLOCK, D_ADAPT_C)
    open_k    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tape_mask = cv2.morphologyEx(tape_mask, cv2.MORPH_OPEN, open_k)
    blurred   = cv2.GaussianBlur(enhanced, (5, 5), 0)
    canny     = cv2.Canny(blurred, D_CANNY_LOW, D_CANNY_HIGH)
    edges_gated = cv2.bitwise_and(canny, tape_mask)
    return enhanced, edges_gated, tape_mask


def dash_filter_contours(roi_mask: np.ndarray) -> np.ndarray:
    """
    v1 contour filter: morphological close + dilate, then
    area / elongation / solidity checks.
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
        _, (rw, rh), _ = cv2.minAreaRect(cnt)
        long_s  = max(rw, rh)
        short_s = max(min(rw, rh), 1e-3)
        if long_s / short_s < D_MIN_ELONGATION:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0 or (area / hull_area) < D_MIN_SOLIDITY:
            continue
        cv2.drawContours(out, [cnt], -1, 255, -1)
    return out


def dash_find_center_x(filtered: np.ndarray,
                        last_cx,
                        recovery: bool,
                        seed_cx=None):
    """
    v1 histogram search: find the dominant column of dash pixels.
    Priority: last_cx → tight seed → wide seed → full blind search.
    """
    hist = np.sum(filtered, axis=0).astype(np.float32)

    if last_cx is not None:
        window = D_SEARCH_WINDOW_RECOVERY if recovery else D_SEARCH_WINDOW
        lo = max(0, last_cx - window)
        hi = min(UNWARPED_SIZE[0], last_cx + window)
        w2 = np.zeros_like(hist); w2[lo:hi] = hist[lo:hi]
        if w2.max() >= 5:
            return int(np.argmax(w2))

    if seed_cx is not None:
        lo = max(0, seed_cx - D_SEARCH_WINDOW_SEED)
        hi = min(UNWARPED_SIZE[0], seed_cx + D_SEARCH_WINDOW_SEED)
        w2 = np.zeros_like(hist); w2[lo:hi] = hist[lo:hi]
        if w2.max() >= 5:
            return int(np.argmax(w2))
        lo = max(0, seed_cx - D_SEARCH_WINDOW_RECOVERY)
        hi = min(UNWARPED_SIZE[0], seed_cx + D_SEARCH_WINDOW_RECOVERY)
        w2 = np.zeros_like(hist); w2[lo:hi] = hist[lo:hi]
        if w2.max() >= 5:
            return int(np.argmax(w2))

    if hist.max() >= 5:
        return int(np.argmax(hist))

    return last_cx if last_cx is not None else seed_cx


def dash_fit(filtered: np.ndarray,
             center_x,
             recovery: bool,
             seed_cx=None):
    """
    v1 window-lock → 2nd-degree polyfit on dash pixels.
    Returns (coeffs | None, ys, xs).
    """
    ref_cx = center_x
    if ref_cx is None:
        ref_cx = seed_cx

    if ref_cx is None:
        locked = filtered
    else:
        window = D_SEARCH_WINDOW_RECOVERY if recovery else D_SEARCH_WINDOW
        lo = max(0, ref_cx - window)
        hi = min(UNWARPED_SIZE[0], ref_cx + window)
        locked = np.zeros_like(filtered)
        locked[:, lo:hi] = filtered[:, lo:hi]

    ys, xs = np.where(locked > 0)
    if len(ys) < 20:
        return None, ys, xs

    coeffs = np.polyfit(ys, xs, 2)
    return coeffs, ys, xs


def dash_is_valid(new_c: np.ndarray, history: deque, recovery: bool = False):
    """
    Validate a new dash fit against the smoothed history.
    Returns (valid, lat_jump, curv_jump).
    """
    if len(history) == 0:
        return True, 0.0, 0.0
    last      = np.mean(history, axis=0)
    lat_jump  = abs(np.polyval(new_c, UNWARPED_SIZE[1] - 1)
                    - np.polyval(last, UNWARPED_SIZE[1] - 1))
    curv_jump = abs(new_c[0] - last[0])
    lat_limit = D_MAX_LAT_JUMP * 3 if recovery else D_MAX_LAT_JUMP
    valid     = (lat_jump  <= lat_limit and
                 curv_jump <= D_MAX_CURV_JUMP)
    return valid, float(lat_jump), float(curv_jump)


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def draw_metrics_panel(img: np.ndarray, m: dict) -> np.ndarray:
    panel   = img.copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (560, 450), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)

    C_OK   = (0, 255, 0)
    C_WARN = (0, 165, 255)
    C_BAD  = (0, 0, 255)
    C_INFO = (255, 255, 255)
    C_YEL  = (0, 220, 220)
    C_DASH = (200, 200, 100)
    C_LLC  = (255, 0, 255)

    def put(t, y, c=C_INFO):
        cv2.putText(panel, t, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, c, 1)

    sc = (C_OK   if m['status'].startswith('OK')  else
          C_WARN if m['status'].startswith('REJ') else C_BAD)

    put(f"Status       : {m['status']}",        26, sc)
    put(f"Frame        : {m.get('frame', 0):6d}", 50, C_INFO)

    put("-- YELLOW (primary left edge) --------------",  78, C_YEL)
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

    put("-- DASH  (secondary center) ----------------", 172, C_DASH)
    put(f"  mode : {m['d_mode']}   pts : {m['pts_dash']:3d}   "
        f"recovery : {'YES' if m['d_recovery'] else 'no'}",
        194,
        C_OK if m['d_mode'] == 'DETECT' else C_WARN)
    put(f"  A={m['d_A']:+.6f}  B={m['d_B']:+.4f}", 216, C_DASH)
    put(f"  lat_jump={m['d_lat']:.2f}px (max {D_MAX_LAT_JUMP})  "
        f"curv_jump={m['d_curv']:.5f} (max {D_MAX_CURV_JUMP})",
        238,
        C_OK if m['d_lat'] < D_MAX_LAT_JUMP else C_BAD)

    put("-- RIGHT (offset from center) --------------", 266, C_INFO)
    put(f"  L2C={L2C_OFFSET}px   C2R={C2R_OFFSET}px", 288, C_INFO)

    put("-- LEFT LANE CENTER  (mid yellow<->dash) ----", 316, C_LLC)
    if m.get('llc_ok', False):
        err = m.get('pixel_error', 0.0)
        direction = ('vehicle LEFT of lane-center -> steer RIGHT' if err < 0
                     else ('vehicle RIGHT of lane-center -> steer LEFT'
                           if err > 0 else 'centered'))
        put(f"  lane_center_x={m['llc_bottom']:.1f}px   "
            f"cam_center_x={m['cam_center_x']:.1f}px",
            338, C_LLC)
        put(f"  pixel_error={err:+.1f}px   ({direction})",
            360, C_OK if abs(err) < 15 else C_WARN)
    else:
        put("  unavailable — need both yellow + dash/offset center",
            338, C_BAD)

    put(f"Y: HSV[{Y_HSV_LOW[0]}-{Y_HSV_HIGH[0]}] "
        f"smooth={Y_SMOOTH_N} rec_thr={Y_RECOVERY_THR}",
        388, C_YEL)
    put(f"D: adaptB={D_ADAPT_BLOCK} C={D_ADAPT_C} "
        f"smooth={D_SMOOTH_N} rec_thr={D_RECOVERY_THR}",
        410, C_DASH)
    put(f"D: area=[{D_MIN_AREA},{D_MAX_AREA}] "
        f"elong={D_MIN_ELONGATION} sol={D_MIN_SOLIDITY}",
        432, C_DASH)

    return panel


def build_fit_vis(ys_y, xs_y, ys_d, xs_d,
                  left_x, center_x, right_x,
                  ploty, l_valid, c_valid, r_valid,
                  left_lane_center_x=None, llc_valid=None,
                  cam_center_x=None, pixel_error=None) -> np.ndarray:
    """
    Bird's-eye visualisation of raw detection pixels and fitted curves.
    """
    vis = np.zeros((UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)

    if len(ys_y) > 0:
        ys_y_c = np.clip(ys_y, 0, UNWARPED_SIZE[1] - 1)
        xs_y_c = np.clip(xs_y, 0, UNWARPED_SIZE[0] - 1)
        vis[ys_y_c, xs_y_c] = (0, 215, 255)

    if len(ys_d) > 0:
        ys_d_c = np.clip(ys_d, 0, UNWARPED_SIZE[1] - 1)
        xs_d_c = np.clip(xs_d, 0, UNWARPED_SIZE[0] - 1)
        vis[ys_d_c, xs_d_c] = (200, 200, 200)

    if cam_center_x is not None:
        ccx = int(round(cam_center_x))
        for y in range(0, UNWARPED_SIZE[1], 14):
            cv2.line(vis, (ccx, y), (ccx, min(y + 7, UNWARPED_SIZE[1] - 1)),
                     (255, 255, 255), 1)

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
        if llc_valid is not None and llc_valid[i]:
            mxi = int(left_lane_center_x[i])
            if 0 <= mxi < UNWARPED_SIZE[0]:
                cv2.circle(vis, (mxi, cy), 2, (255, 0, 255), -1)

    if (cam_center_x is not None and pixel_error is not None and
            llc_valid is not None and llc_valid[-1]):
        ccx  = int(round(cam_center_x))
        lly  = int(ploty[-1])
        llcx = int(left_lane_center_x[-1])
        lly  = min(lly, UNWARPED_SIZE[1] - 1)
        cv2.line(vis, (ccx, lly), (llcx, lly), (0, 255, 0), 2)
        cv2.drawMarker(vis, (ccx, lly), (255, 255, 255),
                        cv2.MARKER_TRIANGLE_UP, 10, 2)
        cv2.drawMarker(vis, (llcx, lly), (255, 0, 255),
                        cv2.MARKER_DIAMOND, 10, 2)
        cv2.putText(vis, f"err={pixel_error:+.1f}px",
                    (min(ccx, llcx), max(lly - 10, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    cv2.LINE_AA)

    cv2.putText(vis,
                'Yellow=left  Cyan=center  Magenta=lane-center  '
                'White=cam-centerline',
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    return vis


# ═══════════════════════════════════════════════════════════════════════════
#  STATEFUL LANE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class LaneDetector:
    """
    Yellow left edge  → v2 HSV pipeline,     own history / thresholds
    Center dashes     → v1 adaptive pipeline, own history / thresholds
    Right edge        → always center + C2R_OFFSET
    """

    def __init__(self,
                 l2c: int = L2C_OFFSET,
                 c2r: int = C2R_OFFSET):
        self.l2c = l2c
        self.c2r = c2r

        # ── yellow tracking state (v2) ────────────────────────────────────
        self._y_history  = deque(maxlen=Y_SMOOTH_N)
        self._y_last_x   = None
        self._y_rejects  = 0
        self._y_recovery = False

        # ── dash tracking state (v1) ──────────────────────────────────────
        self._d_history        = deque(maxlen=D_SMOOTH_N)
        self._d_last_cx        = None
        self._d_rejects        = 0
        self._d_recovery       = False
        self._d_consec_rejects = 0

        # ── smoothed center output ────────────────────────────────────────
        self._center_out_history = deque(maxlen=D_OUTPUT_SMOOTH_N)

        self._frame_idx = 0

        self.metrics = {
            'status':     'INIT',
            'frame':      0,
            'l2c':        l2c,
            'c2r':        c2r,
            # yellow
            'y_mode':     'INIT', 'pts_yellow': 0,
            'y_recovery': False,  'y_bot':      0.0,
            'y_A':        0.0,    'y_B':        0.0,
            'y_lat':      0.0,    'y_curv':     0.0,
            # dash
            'd_mode':     'INIT', 'pts_dash':   0,
            'd_recovery': False,
            'd_A':        0.0,    'd_B':        0.0,
            'd_lat':      0.0,    'd_curv':     0.0,
            # left-lane-center / pixel error
            'llc_ok':       False,
            'llc_bottom':   0.0,
            'cam_center_x': CAMERA_CENTER_X,
            'pixel_error':  0.0,
        }

    # ─────────────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray):
        """
        Run the full detection pipeline on one BGR frame.

        Returns
        -------
        overlay_frame : np.ndarray  (ORIGINAL_SIZE BGR)
        dbg           : np.ndarray  (1920×1080 BGR)
        """
        self._frame_idx += 1

        # ── resize / undistort / warp ─────────────────────────────────────
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
         y_bot) = yellow_fit(yellow_filt,
                             self._y_last_x,
                             self._y_recovery)

        y_lat  = 0.0
        y_curv = 0.0
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

        # ── total-pixel gate ─────────────────────────────────────────────
        # Reject frames where the filtered mask is too sparse to support a
        # reliable poly fit.  A single marginal blob (wall edge, reflection)
        # passes shape checks but produces far fewer pixels than two or more
        # real dashes.  Treating sparse frames as LOW_PTS lets the history
        # fallback serve a stable center line instead of locking onto noise.
        _dash_total_pts = int(np.count_nonzero(dash_filt))
        if _dash_total_pts < D_MIN_TOTAL_PTS:
            dash_filt = np.zeros_like(dash_filt)   # force LOW_PTS path

        if left_coeffs is not None:
            y_bot_x = float(np.polyval(left_coeffs, UNWARPED_SIZE[1] - 1))
            dash_seed_cx = int(np.clip(y_bot_x + self.l2c, 0, UNWARPED_SIZE[0] - 1))
        else:
            dash_seed_cx = None

        self._d_last_cx = dash_find_center_x(
            dash_filt, self._d_last_cx, self._d_recovery,
            seed_cx=dash_seed_cx)

        d_coeffs_new, ys_d, xs_d = dash_fit(
            dash_filt, self._d_last_cx, self._d_recovery,
            seed_cx=dash_seed_cx)

        # ── spatial-plausibility guard ───────────────────────────────────
        # The center dashed line must sit to the RIGHT of the yellow tape.
        # If yellow is tracked and the new fit falls left of
        # (yellow_bottom + D_SPATIAL_GUARD), it is a false positive
        # (wall edge, floor reflection, etc.) and is discarded before
        # reaching the history/validation logic.
        if d_coeffs_new is not None and left_coeffs is not None:
            _y_bot = float(np.polyval(left_coeffs,    UNWARPED_SIZE[1] - 1))
            _d_bot = float(np.polyval(d_coeffs_new,   UNWARPED_SIZE[1] - 1))
            if _d_bot < _y_bot + D_SPATIAL_GUARD:
                d_coeffs_new = None   # impossible geometry — treat as no fit

        d_lat  = 0.0
        d_curv = 0.0
        d_mode = 'DETECT'

        if d_coeffs_new is not None:
            valid, d_lat, d_curv = dash_is_valid(
                d_coeffs_new, self._d_history, recovery=self._d_recovery)

            if valid or self._d_recovery:
                if self._d_recovery:
                    self._d_history.clear()
                self._d_history.append(d_coeffs_new)
                self._d_rejects        = 0
                self._d_consec_rejects = 0
                self._d_recovery       = False
            else:
                self._d_rejects        += 1
                self._d_consec_rejects += 1
                if self._d_rejects >= D_RECOVERY_THR:
                    self._d_recovery       = True
                    self._d_last_cx        = None
                    self._d_consec_rejects = 0
                d_mode = 'REJ'
        else:
            self._d_rejects        += 1
            self._d_consec_rejects += 1
            if self._d_rejects >= D_RECOVERY_THR:
                self._d_recovery       = True
                self._d_last_cx        = None
                self._d_consec_rejects = 0
            d_mode = 'LOW_PTS'

        if (self._d_history and
                self._d_consec_rejects < D_OFFSET_FALLBACK_THR):
            center_coeffs = np.mean(self._d_history, axis=0)
        else:
            center_coeffs = None
            if d_mode == 'DETECT':
                d_mode = 'OFFSET'

        pts_dash = len(ys_d)

        if center_coeffs is None and left_coeffs is not None:
            candidate    = left_coeffs.copy()
            candidate[2] = left_coeffs[2] + self.l2c
            d_mode       = 'OFFSET'
        elif center_coeffs is not None:
            candidate = center_coeffs
        else:
            candidate = None

        if candidate is not None:
            if len(self._center_out_history) > 0:
                last_out   = np.mean(self._center_out_history, axis=0)
                last_bot   = np.polyval(last_out,  UNWARPED_SIZE[1] - 1)
                cand_bot   = np.polyval(candidate, UNWARPED_SIZE[1] - 1)
                jump       = cand_bot - last_bot
                if abs(jump) <= D_OUTPUT_MAX_LAT:
                    self._center_out_history.append(candidate)
                else:
                    direction  = np.sign(jump)
                    target_bot = last_bot + direction * D_OUTPUT_MAX_LAT
                    stepped    = candidate.copy()
                    stepped[2] += target_bot - cand_bot
                    self._center_out_history.append(stepped)
            else:
                self._center_out_history.append(candidate)

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
        #  LEFT LANE CENTER  +  PIXEL ERROR
        # ══════════════════════════════════════════════════════════════════
        left_lane_center_x = (left_x + center_x) / 2.0
        llc_valid = l_valid & c_valid

        valid_idx = np.where(llc_valid)[0]
        if len(valid_idx):
            ref_idx    = valid_idx[-1]
            llc_bottom = float(left_lane_center_x[ref_idx])
            llc_ok     = True
        else:
            llc_bottom = 0.0
            llc_ok     = False

        pixel_error = (CAMERA_CENTER_X - llc_bottom) if llc_ok else 0.0

        # ══════════════════════════════════════════════════════════════════
        #  METRICS
        # ══════════════════════════════════════════════════════════════════
        status = ('OK' if left_coeffs is not None else 'NO_YELLOW')

        self.metrics.update({
            'status':     status,
            'frame':      self._frame_idx,
            'l2c':        self.l2c,
            'c2r':        self.c2r,
            # yellow
            'y_mode':     y_mode,
            'pts_yellow': pts_yellow,
            'y_recovery': self._y_recovery,
            'y_bot':      (float(np.polyval(left_coeffs, UNWARPED_SIZE[1] - 1))
                           if left_coeffs is not None else 0.0),
            'y_A': float(left_coeffs[0])   if left_coeffs   is not None else 0.0,
            'y_B': float(left_coeffs[1])   if left_coeffs   is not None else 0.0,
            'y_lat':  y_lat,
            'y_curv': y_curv,
            # dash
            'd_mode':     d_mode,
            'pts_dash':   pts_dash,
            'd_recovery': self._d_recovery,
            'd_A': float(center_coeffs[0]) if center_coeffs is not None else 0.0,
            'd_B': float(center_coeffs[1]) if center_coeffs is not None else 0.0,
            'd_lat':  d_lat,
            'd_curv': d_curv,
            # left-lane-center / pixel error
            'llc_ok':       llc_ok,
            'llc_bottom':   llc_bottom,
            'cam_center_x': CAMERA_CENTER_X,
            'pixel_error':  pixel_error,
        })

        # ══════════════════════════════════════════════════════════════════
        #  OUTPUTS
        # ══════════════════════════════════════════════════════════════════
        fit_vis = build_fit_vis(
            ys_y, xs_y, ys_d, xs_d,
            left_x, center_x, right_x,
            ploty, l_valid, c_valid, r_valid,
            left_lane_center_x=left_lane_center_x, llc_valid=llc_valid,
            cam_center_x=CAMERA_CENTER_X, pixel_error=pixel_error)

        # ── polygon overlay → unwarp back to camera view ──────────────────
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
        pts_llc = np.array([[int(left_lane_center_x[i]), int(ploty[i])]
                             for i in range(len(ploty)) if llc_valid[i]],
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
        if len(pts_llc) > 1:
            cv2.polylines(bird_ov, [pts_llc], False, (255, 0, 255), 3)

        if llc_ok:
            bot_y  = int(ploty[-1])
            ccx_i  = int(round(CAMERA_CENTER_X))
            llcx_i = int(round(llc_bottom))
            cv2.drawMarker(bird_ov, (ccx_i, bot_y), (255, 255, 255),
                            cv2.MARKER_TRIANGLE_UP, 16, 3)
            cv2.drawMarker(bird_ov, (llcx_i, bot_y), (255, 0, 255),
                            cv2.MARKER_DIAMOND, 16, 3)
            cv2.line(bird_ov, (ccx_i, bot_y), (llcx_i, bot_y),
                     (0, 255, 0), 3)

        overlay_inv   = cv2.warpPerspective(
            bird_ov, M_inv, ORIGINAL_SIZE, flags=cv2.INTER_LINEAR)
        overlay_frame = cv2.addWeighted(undist, 1.0, overlay_inv, 0.5, 0)
        overlay_frame = draw_metrics_panel(overlay_frame, self.metrics)

        dbg = build_debug_mosaic_1080p(
            undist        = undist,
            warped        = warped,
            yellow_filt   = yellow_filt,
            dash_filt     = dash_filt,
            fit_vis       = fit_vis,
            overlay_frame = overlay_frame,
            metrics       = self.metrics
        )

        return overlay_frame, dbg


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO I/O
# ═══════════════════════════════════════════════════════════════════════════

def open_writer(path: str, w: int, h: int, fps: float):
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer


def progress_bar(cur: int, total: int, bar_len: int = 50) -> str:
    filled = int(bar_len * cur / max(total, 1))
    bar    = '█' * filled + '░' * (bar_len - filled)
    return (f"\r[{bar}] {cur:6d}/{total:6d}  "
            f"({100 * cur / max(total, 1):5.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Lane Detection v3 — video inference')
    p.add_argument('--input',       '-i', required=True)
    p.add_argument('--out-debug',   default='debug_grid.mp4')
    p.add_argument('--out-overlay', default='lane_overlay.mp4')
    p.add_argument('--out-csv',     default='pixel_error.csv',
                    help='Per-frame left-lane-center / pixel-error log')
    p.add_argument('--no-display',  action='store_true')
    p.add_argument('--l2c-offset',  type=int, default=L2C_OFFSET)
    p.add_argument('--c2r-offset',  type=int, default=C2R_OFFSET)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--end-frame',   type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Not found: {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.input}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'─'*62}")
    print(f"  Input  : {args.input}  "
          f"[{src_w}x{src_h}  {fps:.2f}fps  {total}fr]")
    print(f"{'─'*62}\n")

    start    = max(0, args.start_frame)
    end      = total if args.end_frame < 0 else min(args.end_frame, total)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    n_frames = end - start

    detector = LaneDetector(l2c=args.l2c_offset, c2r=args.c2r_offset)

    # ── probe output sizes from the first frame ───────────────────────────
    ret, first = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame.")
        sys.exit(1)

    ov_f, dbg_f = detector.process_frame(first.copy())
    ov_h,  ov_w  = ov_f.shape[:2]
    dbg_h, dbg_w = dbg_f.shape[:2]

    assert dbg_w == CANVAS_W and dbg_h == CANVAS_H, \
        f"Unexpected mosaic size {dbg_w}x{dbg_h}"

    try:
        w_ov  = open_writer(args.out_overlay, ov_w,     ov_h,     fps)
        w_dbg = open_writer(args.out_debug,   CANVAS_W, CANVAS_H, fps)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        cap.release()
        sys.exit(1)

    csv_file   = open(args.out_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame', 'status', 'llc_ok',
        'left_lane_center_x_px', 'cam_center_x_px', 'pixel_error_px'
    ])

    def log_csv_row():
        m = detector.metrics
        csv_writer.writerow([
            m.get('frame', 0), m.get('status', ''), m.get('llc_ok', False),
            f"{m.get('llc_bottom', 0.0):.2f}",
            f"{m.get('cam_center_x', CAMERA_CENTER_X):.2f}",
            f"{m.get('pixel_error', 0.0):.2f}",
        ])

    print(f"  overlay   → {args.out_overlay}  [{ov_w}x{ov_h}]")
    print(f"  debug     → {args.out_debug}    [{CANVAS_W}x{CANVAS_H}]")
    print(f"  pixel-err → {args.out_csv}\n")

    w_ov.write(ov_f)
    w_dbg.write(dbg_f)
    log_csv_row()

    SHOW = not args.no_display
    if SHOW:
        PREV_W = 1280
        PREV_H = int(dbg_h * PREV_W / max(dbg_w, 1))
        cv2.namedWindow('Debug',   cv2.WINDOW_NORMAL)
        cv2.namedWindow('Overlay', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug',   PREV_W, PREV_H)
        cv2.resizeWindow('Overlay', 1280, 720)
        cv2.imshow('Debug',   dbg_f)
        cv2.imshow('Overlay', cv2.resize(ov_f, (1280, 720)))
        cv2.waitKey(1)

    for fn in range(1, n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        ov_f, dbg_f = detector.process_frame(frame)
        w_ov.write(ov_f)
        w_dbg.write(dbg_f)
        log_csv_row()

        print(progress_bar(fn, n_frames), end='', flush=True)

        if SHOW:
            cv2.imshow('Debug',   dbg_f)
            cv2.imshow('Overlay', cv2.resize(ov_f, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Quit.")
                break

    print(f"\n\n{'─'*62}")
    print(f"  Done.  {n_frames} frames processed.")
    print(f"  {args.out_overlay}")
    print(f"  {args.out_debug}")
    print(f"  {args.out_csv}")
    print(f"{'─'*62}\n")

    cap.release()
    w_ov.release()
    w_dbg.release()
    csv_file.close()

    if SHOW:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()