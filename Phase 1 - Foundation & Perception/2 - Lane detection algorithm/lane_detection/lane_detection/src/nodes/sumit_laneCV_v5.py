#!/usr/bin/env python3
"""
sumit_laneCV_node.py
====================
ROS 2 node  —  node name: sumit_laneCV

─────────────────────────────────────────────────────────────────────────────
EDIT THESE TWO LINES TO CONFIGURE THE NODE
─────────────────────────────────────────────────────────────────────────────
"""
DEBUG_VIDEO_SAVE_PATH = '/home/sumit/recordings/lane_debug.mp4'   # ← output video
DEBUG_VIDEO_FPS       = 30.0                                    # ← match camera fps
# ─────────────────────────────────────────────────────────────────────────────

"""
Subscribes
----------
/img/raw_rgb   sensor_msgs/Image   raw OAK-D camera feed

Publishes
---------
/lane/debug_image            sensor_msgs/Image          1920×1080 debug mosaic
/lane/pixel_error            std_msgs/Float32           signed lateral error (px)
/lane/polynomial_coeffs      std_msgs/Float64MultiArray 9 coefficients (see below)
/lane/status                 std_msgs/String            one-line status string
/lane/left_lane_centre_path  nav_msgs/Path              LLC waypoints, base_link metres

ROS parameters
--------------
mpx         (float, default 0.005)  metres per pixel, lateral   (column direction)
mpy         (float, default 0.005)  metres per pixel, longitudinal (row direction)
base_frame  (str,   default base_link)
n_waypoints (int,   default 40)

Polynomial coefficients layout — /lane/polynomial_coeffs
---------------------------------------------------------
All polys are 2nd-order in bird's-eye PIXEL space:
    x_col(row) = A·row² + B·row + C
Nine floats in this fixed order:
  [0] y_A   [1] y_B   [2] y_C    ← smoothed yellow left-edge
  [3] d_A   [4] d_B   [5] d_C    ← smoothed centre-dash (or OFFSET estimate)
  [6] llc_A [7] llc_B [8] llc_C  ← left-lane-centre = midpoint of yellow & dash
All three are all-zeros when the respective source is unavailable.

Pixel error sign  (/lane/pixel_error)
--------------------------------------
  pixel_error = CAMERA_CENTER_X − llc_bottom_col
  Positive → vehicle is LEFT  of lane centre → steer RIGHT
  Negative → vehicle is RIGHT of lane centre → steer LEFT
  Always published; 0.0 when lane is lost.

base_link conversion  (for /lane/left_lane_centre_path)
---------------------------------------------------------
  base_x =  (UNWARPED_H − 1 − row) × mpy        (+X forward)
  base_y = −(col − CAMERA_CENTER_X) × mpx        (+Y left, REP-105)
  Waypoints ordered nearest (pose[0]) → farthest (pose[-1]).

Run
---
  python3 sumit_laneCV_node.py
  -- or --
  ros2 run <pkg> sumit_laneCV_node --ros-args -p mpx:=0.002 -p mpy:=0.002
"""

import os
import cv2
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos  import (QoSProfile, QoSReliabilityPolicy,
                         QoSHistoryPolicy, QoSDurabilityPolicy)
from cv_bridge  import CvBridge

from std_msgs.msg      import Float32, Float64MultiArray, String
from sensor_msgs.msg   import Image
from nav_msgs.msg      import Path
from geometry_msgs.msg import PoseStamped


# ═══════════════════════════════════════════════════════════════════════════
#  GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

UNWARPED_SIZE = (500, 600)
ORIGINAL_SIZE = (1920, 1080)

SRC = np.array([
    (675, 543),
    (1247, 517),
    (1883, 851),
    (43,  882),
], dtype=np.float32)

DST = np.array(
    [(0, 0), (500, 0), (500, 600), (0, 600)],
    dtype=np.float32,
)

K = np.array([
    [1459.2276611328125, 0.0,                945.8712768554688],
    [0.0,               1458.7711181640625,  516.719482421875 ],
    [0.0,               0.0,                 1.0              ],
], dtype=np.float64)

D = np.array([
    -2.480555534362793, -6.659245014190674,  0.00015050164074636996,
     0.0006666610715910792, 22.50022315979004, -2.5853612422943115,
    -6.1252665519714355, 21.745553970336914,  0.0, 0.0, 0.0, 0.0,
    -0.0015395419904962182, 0.0015926380874589086,
], dtype=np.float64)

M     = cv2.getPerspectiveTransform(SRC, DST)
M_inv = cv2.getPerspectiveTransform(DST, SRC)

L2C_OFFSET      = 300        # yellow left  → centre dash (bird's-eye px)
C2R_OFFSET      = 310        # centre dash  → right virtual edge (bird's-eye px)
CAMERA_CENTER_X = UNWARPED_SIZE[0] / 2.0   # = 250.0

ROI_TOP   = 220
ROI_LEFT  = 0
ROI_RIGHT = 485

_BOT_ROW = float(UNWARPED_SIZE[1] - 1)    # 599.0


# ═══════════════════════════════════════════════════════════════════════════
#  DEBUG MOSAIC LAYOUT  (1920 × 1080 canvas)
# ═══════════════════════════════════════════════════════════════════════════

CANVAS_W    = 1920
CANVAS_H    = 1080
PAD_OUTER   = 16
PAD_INNER   = 10
HEADER_H    = 40

_AVAIL_W   = CANVAS_W - 2 * PAD_OUTER
_AVAIL_H   = CANVAS_H - 2 * PAD_OUTER - HEADER_H

TOP_N      = 4
TOP_TILE_W = (_AVAIL_W - (TOP_N - 1) * PAD_INNER) // TOP_N
TOP_TILE_H = (_AVAIL_H - PAD_INNER) * 11 // 20

BOT_N      = 2
BOT_TILE_W = (_AVAIL_W - (BOT_N - 1) * PAD_INNER) // BOT_N
BOT_TILE_H = _AVAIL_H - TOP_TILE_H - PAD_INNER

TOP_ROW_Y  = PAD_OUTER + HEADER_H
BOT_ROW_Y  = TOP_ROW_Y + TOP_TILE_H + PAD_INNER

TOP_COL_X  = [PAD_OUTER + i * (TOP_TILE_W + PAD_INNER) for i in range(TOP_N)]
BOT_COL_X  = [PAD_OUTER + i * (BOT_TILE_W + PAD_INNER) for i in range(BOT_N)]

BG_COLOR     = (18,  18,  18)
HEADER_COLOR = (30,  30,  30)
BORDER_COLOR = (55,  55,  55)
LABEL_BG     = (0,   0,   0)
LABEL_FG     = (0,  220, 220)


# ═══════════════════════════════════════════════════════════════════════════
#  DEBUG MOSAIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _make_tile(img, tw, th, label, metrics=None):
    tile = cv2.resize(_to_bgr(img), (tw, th), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), BORDER_COLOR, 1)
    cv2.rectangle(tile, (0, 0), (tw, 24), LABEL_BG, -1)
    cv2.putText(tile, label, (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, LABEL_FG, 1, cv2.LINE_AA)
    if metrics is not None:
        dot = ((0, 210, 0) if metrics['status'].startswith('OK')
               else (0, 60, 220))
        cv2.circle(tile, (tw - 14, 12), 6, dot, -1)
    return tile


def _paste(canvas, tile, x, y):
    th, tw = tile.shape[:2]
    canvas[y:y + th, x:x + tw] = tile


def build_debug_mosaic_1080p(undist, warped, yellow_filt, dash_filt,
                              fit_vis, overlay_frame, metrics):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0),
                  (CANVAS_W, PAD_OUTER + HEADER_H - PAD_INNER // 2),
                  HEADER_COLOR, -1)
    dot = ((0, 210, 0) if metrics['status'].startswith('OK')
           else (0, 60, 220))
    cv2.circle(canvas, (PAD_OUTER + 14, PAD_OUTER + HEADER_H // 2), 9, dot, -1)
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
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1, cv2.LINE_AA)

    tw, th = TOP_TILE_W, TOP_TILE_H
    bw, bh = BOT_TILE_W, BOT_TILE_H

    _paste(canvas, _make_tile(undist, tw, th, "1: Undistorted", metrics),
           TOP_COL_X[0], TOP_ROW_Y)

    warped_roi = warped.copy()
    cv2.rectangle(warped_roi, (ROI_LEFT, ROI_TOP),
                  (ROI_RIGHT, UNWARPED_SIZE[1] - 1), (0, 220, 220), 2)
    cv2.putText(warped_roi, 'ROI', (ROI_LEFT + 4, ROI_TOP + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)
    _paste(canvas, _make_tile(warped_roi, tw, th, "2: Bird's-Eye + ROI"),
           TOP_COL_X[1], TOP_ROW_Y)

    yf = np.zeros((*yellow_filt.shape, 3), dtype=np.uint8)
    yf[yellow_filt > 0] = (0, 215, 255)
    _paste(canvas, _make_tile(yf, tw, th, "3: Yellow Tape Filtered"),
           TOP_COL_X[2], TOP_ROW_Y)

    df = np.zeros((*dash_filt.shape, 3), dtype=np.uint8)
    df[dash_filt > 0] = (200, 200, 200)
    _paste(canvas, _make_tile(df, tw, th, "4: Dash Lines Filtered"),
           TOP_COL_X[3], TOP_ROW_Y)

    _paste(canvas,
           _make_tile(fit_vis, bw, bh,
                      "5: Poly Fit  Yellow=left  Cyan=center  Red=right"),
           BOT_COL_X[0], BOT_ROW_Y)
    _paste(canvas,
           _make_tile(overlay_frame, bw, bh, "6: Lane Overlay", metrics),
           BOT_COL_X[1], BOT_ROW_Y)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════
#  YELLOW TAPE CONFIG  (v2)
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
#  CENTER DASH CONFIG  (v1)
# ═══════════════════════════════════════════════════════════════════════════

D_CLAHE_CLIP  = 1.5
D_CLAHE_GRID  = (8, 8)
D_ADAPT_BLOCK = 199
D_ADAPT_C     = -45
D_CANNY_LOW   = 215
D_CANNY_HIGH  = 170
D_MIN_AREA       = 120
D_MAX_AREA       = 2000
D_MIN_ELONGATION = 1.5
D_MIN_SOLIDITY   = 0.55
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

_DASH_CLAHE = cv2.createCLAHE(clipLimit=D_CLAHE_CLIP, tileGridSize=D_CLAHE_GRID)


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — YELLOW  (v2)
# ═══════════════════════════════════════════════════════════════════════════

def yellow_extract_mask(bgr_warped):
    hsv  = cv2.cvtColor(bgr_warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, Y_HSV_LOW, Y_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, Y_OPEN_K))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, Y_CLOSE_K))
    return mask


def yellow_filter_contours(mask):
    out = np.zeros_like(mask)
    for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) >= Y_MIN_AREA:
            cv2.drawContours(out, [cnt], -1, 255, -1)
    return out


def yellow_fit(filtered, last_x, recovery):
    hist   = np.sum(filtered, axis=0).astype(np.float32)
    window = Y_SEARCH_WINDOW_RECOVERY if recovery else Y_SEARCH_WINDOW
    if last_x is not None:
        lo = max(0, int(last_x) - window)
        hi = min(UNWARPED_SIZE[0], int(last_x) + window)
        w2 = np.zeros_like(hist); w2[lo:hi] = hist[lo:hi]
        if w2.max() >= 5:
            hist = w2
    if hist.max() < 5:
        return None, np.array([]), np.array([]), last_x
    peak   = int(np.argmax(hist))
    lo     = max(0, peak - window)
    hi     = min(UNWARPED_SIZE[0], peak + window)
    locked = np.zeros_like(filtered); locked[:, lo:hi] = filtered[:, lo:hi]
    ys, xs = np.where(locked > 0)
    if len(ys) < Y_MIN_PTS:
        return None, ys, xs, last_x
    coeffs = np.polyfit(ys, xs, 2)
    bot    = float(np.polyval(coeffs, UNWARPED_SIZE[1] - 1))
    return coeffs, ys, xs, bot


def yellow_is_valid(new_c, history):
    if len(history) == 0:
        return True, 0.0, 0.0
    last      = np.mean(history, axis=0)
    lat_jump  = abs(np.polyval(new_c, UNWARPED_SIZE[1] - 1)
                    - np.polyval(last, UNWARPED_SIZE[1] - 1))
    curv_jump = abs(new_c[0] - last[0])
    return (lat_jump <= Y_MAX_LAT_JUMP and curv_jump <= Y_MAX_CURV_JUMP,
            float(lat_jump), float(curv_jump))


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — CENTRE DASHES  (v1)
# ═══════════════════════════════════════════════════════════════════════════

def dash_preprocess(gray):
    enhanced    = _DASH_CLAHE.apply(gray)
    tape_mask   = cv2.adaptiveThreshold(enhanced, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY,
                                        D_ADAPT_BLOCK, D_ADAPT_C)
    tape_mask   = cv2.morphologyEx(tape_mask, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    blurred     = cv2.GaussianBlur(enhanced, (5, 5), 0)
    canny       = cv2.Canny(blurred, D_CANNY_LOW, D_CANNY_HIGH)
    edges_gated = cv2.bitwise_and(canny, tape_mask)
    return enhanced, edges_gated, tape_mask


def dash_filter_contours(roi_mask):
    closed  = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    dilated = cv2.dilate(closed,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                         iterations=2)
    out = np.zeros_like(dilated)
    for cnt in cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if not (D_MIN_AREA <= area <= D_MAX_AREA):
            continue
        _, (rw, rh), _ = cv2.minAreaRect(cnt)
        if max(rw, rh) / max(min(rw, rh), 1e-3) < D_MIN_ELONGATION:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0 or (area / hull_area) < D_MIN_SOLIDITY:
            continue
        cv2.drawContours(out, [cnt], -1, 255, -1)
    return out


def dash_find_center_x(filtered, last_cx, recovery, seed_cx=None):
    hist = np.sum(filtered, axis=0).astype(np.float32)
    if last_cx is not None:
        window = D_SEARCH_WINDOW_RECOVERY if recovery else D_SEARCH_WINDOW
        lo = max(0, last_cx - window); hi = min(UNWARPED_SIZE[0], last_cx + window)
        w2 = np.zeros_like(hist); w2[lo:hi] = hist[lo:hi]
        if w2.max() >= 5:
            return int(np.argmax(w2))
    if seed_cx is not None:
        for win in (D_SEARCH_WINDOW_SEED, D_SEARCH_WINDOW_RECOVERY):
            lo = max(0, seed_cx - win); hi = min(UNWARPED_SIZE[0], seed_cx + win)
            w2 = np.zeros_like(hist); w2[lo:hi] = hist[lo:hi]
            if w2.max() >= 5:
                return int(np.argmax(w2))
    if hist.max() >= 5:
        return int(np.argmax(hist))
    return last_cx if last_cx is not None else seed_cx


def dash_fit(filtered, center_x, recovery, seed_cx=None):
    ref_cx = center_x if center_x is not None else seed_cx
    if ref_cx is None:
        locked = filtered
    else:
        window = D_SEARCH_WINDOW_RECOVERY if recovery else D_SEARCH_WINDOW
        lo = max(0, ref_cx - window); hi = min(UNWARPED_SIZE[0], ref_cx + window)
        locked = np.zeros_like(filtered); locked[:, lo:hi] = filtered[:, lo:hi]
    ys, xs = np.where(locked > 0)
    if len(ys) < 20:
        return None, ys, xs
    return np.polyfit(ys, xs, 2), ys, xs


def dash_is_valid(new_c, history, recovery=False):
    if len(history) == 0:
        return True, 0.0, 0.0
    last      = np.mean(history, axis=0)
    lat_jump  = abs(np.polyval(new_c, UNWARPED_SIZE[1] - 1)
                    - np.polyval(last, UNWARPED_SIZE[1] - 1))
    curv_jump = abs(new_c[0] - last[0])
    lat_limit = D_MAX_LAT_JUMP * 3 if recovery else D_MAX_LAT_JUMP
    return (lat_jump <= lat_limit and curv_jump <= D_MAX_CURV_JUMP,
            float(lat_jump), float(curv_jump))


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def draw_metrics_panel(img, m):
    panel   = img.copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (560, 450), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)

    C_OK   = (0, 255, 0);  C_WARN = (0, 165, 255); C_BAD  = (0, 0, 255)
    C_INFO = (255, 255, 255)
    C_YEL  = (0, 220, 220); C_DASH = (200, 200, 100); C_LLC = (255, 0, 255)

    def put(t, y, c=C_INFO):
        cv2.putText(panel, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, c, 1)

    sc = (C_OK   if m['status'].startswith('OK')  else
          C_WARN if m['status'].startswith('REJ') else C_BAD)
    put(f"Status       : {m['status']}",         26, sc)
    put(f"Frame        : {m.get('frame', 0):6d}", 50)
    put("-- YELLOW (primary left edge) --------------", 78, C_YEL)
    put(f"  mode : {m['y_mode']}   pts : {m['pts_yellow']:3d}   "
        f"recovery : {'YES' if m['y_recovery'] else 'no'}",
        100, C_OK if m['pts_yellow'] >= Y_MIN_PTS else C_BAD)
    put(f"  A={m['y_A']:+.6f}  B={m['y_B']:+.4f}  "
        f"left_x_bot={m['y_bot']:.1f}px", 122, C_YEL)
    put(f"  lat_jump={m['y_lat']:.2f}px (max {Y_MAX_LAT_JUMP})  "
        f"curv_jump={m['y_curv']:.5f} (max {Y_MAX_CURV_JUMP})",
        144, C_OK if m['y_lat'] < Y_MAX_LAT_JUMP else C_BAD)
    put("-- DASH  (secondary center) ----------------", 172, C_DASH)
    put(f"  mode : {m['d_mode']}   pts : {m['pts_dash']:3d}   "
        f"recovery : {'YES' if m['d_recovery'] else 'no'}",
        194, C_OK if m['d_mode'] == 'DETECT' else C_WARN)
    put(f"  A={m['d_A']:+.6f}  B={m['d_B']:+.4f}", 216, C_DASH)
    put(f"  lat_jump={m['d_lat']:.2f}px (max {D_MAX_LAT_JUMP})  "
        f"curv_jump={m['d_curv']:.5f} (max {D_MAX_CURV_JUMP})",
        238, C_OK if m['d_lat'] < D_MAX_LAT_JUMP else C_BAD)
    put("-- RIGHT (offset from center) --------------", 266)
    put(f"  L2C={L2C_OFFSET}px   C2R={C2R_OFFSET}px", 288)
    put("-- LEFT LANE CENTER  (mid yellow<->dash) ----", 316, C_LLC)
    if m.get('llc_ok', False):
        err = m.get('pixel_error', 0.0)
        direction = ('vehicle LEFT of lane-center -> steer RIGHT' if err < 0
                     else ('vehicle RIGHT of lane-center -> steer LEFT'
                           if err > 0 else 'centered'))
        put(f"  lane_center_x={m['llc_bottom']:.1f}px   "
            f"cam_center_x={m['cam_center_x']:.1f}px", 338, C_LLC)
        put(f"  pixel_error={err:+.1f}px   ({direction})",
            360, C_OK if abs(err) < 15 else C_WARN)
    else:
        put("  unavailable — need both yellow + dash/offset center", 338, C_BAD)
    put(f"Y: HSV[{Y_HSV_LOW[0]}-{Y_HSV_HIGH[0]}] "
        f"smooth={Y_SMOOTH_N} rec_thr={Y_RECOVERY_THR}", 388, C_YEL)
    put(f"D: adaptB={D_ADAPT_BLOCK} C={D_ADAPT_C} "
        f"smooth={D_SMOOTH_N} rec_thr={D_RECOVERY_THR}", 410, C_DASH)
    put(f"D: area=[{D_MIN_AREA},{D_MAX_AREA}] "
        f"elong={D_MIN_ELONGATION} sol={D_MIN_SOLIDITY}", 432, C_DASH)
    return panel


def build_fit_vis(ys_y, xs_y, ys_d, xs_d,
                  left_x, center_x, right_x,
                  ploty, l_valid, c_valid, r_valid,
                  left_lane_center_x=None, llc_valid=None,
                  cam_center_x=None, pixel_error=None):
    vis = np.zeros((UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)
    if len(ys_y):
        vis[np.clip(ys_y, 0, UNWARPED_SIZE[1]-1),
            np.clip(xs_y, 0, UNWARPED_SIZE[0]-1)] = (0, 215, 255)
    if len(ys_d):
        vis[np.clip(ys_d, 0, UNWARPED_SIZE[1]-1),
            np.clip(xs_d, 0, UNWARPED_SIZE[0]-1)] = (200, 200, 200)
    if cam_center_x is not None:
        ccx = int(round(cam_center_x))
        for y in range(0, UNWARPED_SIZE[1], 14):
            cv2.line(vis, (ccx, y), (ccx, min(y+7, UNWARPED_SIZE[1]-1)),
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
        if llc_valid is not None and llc_valid[i] and left_lane_center_x is not None:
            mxi = int(left_lane_center_x[i])
            if 0 <= mxi < UNWARPED_SIZE[0]:
                cv2.circle(vis, (mxi, cy), 2, (255, 0, 255), -1)
    if (cam_center_x is not None and pixel_error is not None
            and llc_valid is not None and llc_valid[-1]):
        ccx  = int(round(cam_center_x))
        lly  = min(int(ploty[-1]), UNWARPED_SIZE[1] - 1)
        llcx = int(left_lane_center_x[-1])
        cv2.line(vis, (ccx, lly), (llcx, lly), (0, 255, 0), 2)
        cv2.drawMarker(vis, (ccx,  lly), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 10, 2)
        cv2.drawMarker(vis, (llcx, lly), (255, 0, 255),   cv2.MARKER_DIAMOND,     10, 2)
        cv2.putText(vis, f"err={pixel_error:+.1f}px",
                    (min(ccx, llcx), max(lly - 10, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(vis,
                'Yellow=left  Cyan=center  Magenta=lane-center  White=cam-centerline',
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    return vis


# ═══════════════════════════════════════════════════════════════════════════
#  STATEFUL LANE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class LaneDetector:

    def __init__(self, l2c=L2C_OFFSET, c2r=C2R_OFFSET):
        self.l2c = l2c
        self.c2r = c2r
        self._y_history          = deque(maxlen=Y_SMOOTH_N)
        self._y_last_x           = None
        self._y_rejects          = 0
        self._y_recovery         = False
        self._d_history          = deque(maxlen=D_SMOOTH_N)
        self._d_last_cx          = None
        self._d_rejects          = 0
        self._d_recovery         = False
        self._d_consec_rejects   = 0
        self._center_out_history = deque(maxlen=D_OUTPUT_SMOOTH_N)
        self._frame_idx          = 0
        self.metrics = {
            'status': 'INIT', 'frame': 0, 'l2c': l2c, 'c2r': c2r,
            'y_mode': 'INIT', 'pts_yellow': 0,
            'y_recovery': False, 'y_bot': 0.0, 'y_A': 0.0, 'y_B': 0.0,
            'y_lat': 0.0, 'y_curv': 0.0,
            'd_mode': 'INIT', 'pts_dash': 0,
            'd_recovery': False, 'd_A': 0.0, 'd_B': 0.0,
            'd_lat': 0.0, 'd_curv': 0.0,
            'llc_ok': False, 'llc_bottom': 0.0,
            'cam_center_x': CAMERA_CENTER_X, 'pixel_error': 0.0,
        }

    def process_frame(self, frame):
        self._frame_idx += 1

        # undistort + warp
        if (frame.shape[1], frame.shape[0]) != ORIGINAL_SIZE:
            frame = cv2.resize(frame, ORIGINAL_SIZE)
        undist = cv2.undistort(frame, K, D)
        warped = cv2.warpPerspective(undist, M, UNWARPED_SIZE,
                                     flags=cv2.INTER_LINEAR)
        gray   = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # ── yellow pipeline ───────────────────────────────────────────────
        yellow_mask = yellow_extract_mask(warped)
        yellow_roi  = yellow_mask.copy();  yellow_roi[:ROI_TOP, :] = 0
        yellow_filt = yellow_filter_contours(yellow_roi)
        y_coeffs_new, ys_y, xs_y, y_bot = yellow_fit(
            yellow_filt, self._y_last_x, self._y_recovery)

        y_lat = y_curv = 0.0;  y_mode = 'DETECT'
        if y_coeffs_new is not None:
            valid, y_lat, y_curv = yellow_is_valid(y_coeffs_new, self._y_history)
            if valid or self._y_recovery:
                self._y_history.append(y_coeffs_new)
                self._y_last_x = y_bot;  self._y_rejects = 0;  self._y_recovery = False
            else:
                self._y_rejects += 1
                if self._y_rejects >= Y_RECOVERY_THR:
                    self._y_recovery = True;  self._y_last_x = None
                y_mode = 'REJ'
        else:
            self._y_rejects += 1
            if self._y_rejects >= Y_RECOVERY_THR:
                self._y_recovery = True;  self._y_last_x = None
            y_mode = 'LOW_PTS'

        left_coeffs = np.mean(self._y_history, axis=0) if self._y_history else None
        pts_yellow  = len(ys_y)

        # ── dash pipeline ─────────────────────────────────────────────────
        _, edges_gated, _ = dash_preprocess(gray)
        dash_roi = edges_gated.copy()
        dash_roi[:ROI_TOP, :] = 0;  dash_roi[:, :ROI_LEFT] = 0;  dash_roi[:, ROI_RIGHT:] = 0
        dash_filt = dash_filter_contours(dash_roi)

        if left_coeffs is not None:
            y_bot_x      = float(np.polyval(left_coeffs, UNWARPED_SIZE[1] - 1))
            dash_seed_cx = int(np.clip(y_bot_x + self.l2c, 0, UNWARPED_SIZE[0] - 1))
        else:
            dash_seed_cx = None

        self._d_last_cx = dash_find_center_x(
            dash_filt, self._d_last_cx, self._d_recovery, seed_cx=dash_seed_cx)
        d_coeffs_new, ys_d, xs_d = dash_fit(
            dash_filt, self._d_last_cx, self._d_recovery, seed_cx=dash_seed_cx)

        d_lat = d_curv = 0.0;  d_mode = 'DETECT'
        if d_coeffs_new is not None:
            valid, d_lat, d_curv = dash_is_valid(
                d_coeffs_new, self._d_history, recovery=self._d_recovery)
            if valid or self._d_recovery:
                if self._d_recovery:
                    self._d_history.clear()
                self._d_history.append(d_coeffs_new)
                self._d_rejects = self._d_consec_rejects = 0;  self._d_recovery = False
            else:
                self._d_rejects += 1;  self._d_consec_rejects += 1
                if self._d_rejects >= D_RECOVERY_THR:
                    self._d_recovery = True;  self._d_last_cx = None
                    self._d_consec_rejects = 0
                d_mode = 'REJ'
        else:
            self._d_rejects += 1;  self._d_consec_rejects += 1
            if self._d_rejects >= D_RECOVERY_THR:
                self._d_recovery = True;  self._d_last_cx = None
                self._d_consec_rejects = 0
            d_mode = 'LOW_PTS'

        if self._d_history and self._d_consec_rejects < D_OFFSET_FALLBACK_THR:
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
            if self._center_out_history:
                last_out   = np.mean(self._center_out_history, axis=0)
                last_bot   = np.polyval(last_out,  UNWARPED_SIZE[1] - 1)
                cand_bot   = np.polyval(candidate, UNWARPED_SIZE[1] - 1)
                jump       = cand_bot - last_bot
                if abs(jump) <= D_OUTPUT_MAX_LAT:
                    self._center_out_history.append(candidate)
                else:
                    stepped    = candidate.copy()
                    stepped[2] += (last_bot + np.sign(jump) * D_OUTPUT_MAX_LAT) - cand_bot
                    self._center_out_history.append(stepped)
            else:
                self._center_out_history.append(candidate)

        center_coeffs = (np.mean(self._center_out_history, axis=0)
                         if self._center_out_history else None)

        # ── sample curves ──────────────────────────────────────────────────
        ploty    = np.linspace(ROI_TOP, UNWARPED_SIZE[1] - 1, 200)
        left_x   = (np.polyval(left_coeffs,   ploty) if left_coeffs   is not None
                    else np.full_like(ploty, 0.0))
        center_x = (np.polyval(center_coeffs, ploty) if center_coeffs is not None
                    else left_x + self.l2c)
        right_x  = center_x + self.c2r
        l_valid  = (left_x   >= 0) & (left_x   < UNWARPED_SIZE[0])
        c_valid  = (center_x >= 0) & (center_x < UNWARPED_SIZE[0])
        r_valid  = (right_x  >= 0) & (right_x  < UNWARPED_SIZE[0])

        # ── left-lane-centre + pixel error ─────────────────────────────────
        left_lane_center_x = (left_x + center_x) / 2.0
        llc_valid  = l_valid & c_valid
        valid_idx  = np.where(llc_valid)[0]
        if len(valid_idx):
            llc_bottom = float(left_lane_center_x[valid_idx[-1]])
            llc_ok     = True
        else:
            llc_bottom = 0.0;  llc_ok = False

        # pixel_error = CAMERA_CENTER_X - llc_bottom
        # positive -> vehicle LEFT  of lane centre -> steer RIGHT
        # negative -> vehicle RIGHT of lane centre -> steer LEFT
        pixel_error = (CAMERA_CENTER_X - llc_bottom) if llc_ok else 0.0

        # ── metrics ────────────────────────────────────────────────────────
        self.metrics.update({
            'status': 'OK' if left_coeffs is not None else 'NO_YELLOW',
            'frame':  self._frame_idx,
            'l2c': self.l2c, 'c2r': self.c2r,
            'y_mode': y_mode, 'pts_yellow': pts_yellow,
            'y_recovery': self._y_recovery,
            'y_bot': (float(np.polyval(left_coeffs, UNWARPED_SIZE[1] - 1))
                      if left_coeffs is not None else 0.0),
            'y_A': float(left_coeffs[0])   if left_coeffs   is not None else 0.0,
            'y_B': float(left_coeffs[1])   if left_coeffs   is not None else 0.0,
            'y_lat': y_lat, 'y_curv': y_curv,
            'd_mode': d_mode, 'pts_dash': pts_dash,
            'd_recovery': self._d_recovery,
            'd_A': float(center_coeffs[0]) if center_coeffs is not None else 0.0,
            'd_B': float(center_coeffs[1]) if center_coeffs is not None else 0.0,
            'd_lat': d_lat, 'd_curv': d_curv,
            'llc_ok': llc_ok, 'llc_bottom': llc_bottom,
            'cam_center_x': CAMERA_CENTER_X, 'pixel_error': pixel_error,
        })

        # ── build visuals ──────────────────────────────────────────────────
        fit_vis = build_fit_vis(
            ys_y, xs_y, ys_d, xs_d,
            left_x, center_x, right_x, ploty, l_valid, c_valid, r_valid,
            left_lane_center_x=left_lane_center_x, llc_valid=llc_valid,
            cam_center_x=CAMERA_CENTER_X, pixel_error=pixel_error)

        bird_ov = np.zeros((UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)
        pts_l   = np.array([[int(left_x[i]),   int(ploty[i])]
                            for i in range(len(ploty)) if l_valid[i]], dtype=np.int32)
        pts_c   = np.array([[int(center_x[i]), int(ploty[i])]
                            for i in range(len(ploty)) if c_valid[i]], dtype=np.int32)
        pts_r   = np.array([[int(right_x[i]),  int(ploty[i])]
                            for i in range(len(ploty)) if r_valid[i]], dtype=np.int32)
        pts_llc = np.array([[int(left_lane_center_x[i]), int(ploty[i])]
                            for i in range(len(ploty)) if llc_valid[i]], dtype=np.int32)

        if len(pts_l) > 2 and len(pts_r) > 2:
            cv2.fillPoly(bird_ov, [np.concatenate([pts_l, np.flipud(pts_r)])],
                         (0, 180, 0))
        if len(pts_l)   > 1: cv2.polylines(bird_ov, [pts_l],   False, (0, 215, 255), 5)
        if len(pts_c)   > 1: cv2.polylines(bird_ov, [pts_c],   False, (255, 255, 0), 3)
        if len(pts_r)   > 1: cv2.polylines(bird_ov, [pts_r],   False, (0, 0, 255),   3)
        if len(pts_llc) > 1: cv2.polylines(bird_ov, [pts_llc], False, (255, 0, 255), 3)

        if llc_ok:
            bot_y  = int(ploty[-1])
            ccx_i  = int(round(CAMERA_CENTER_X))
            llcx_i = int(round(llc_bottom))
            cv2.drawMarker(bird_ov, (ccx_i,  bot_y), (255, 255, 255),
                           cv2.MARKER_TRIANGLE_UP, 16, 3)
            cv2.drawMarker(bird_ov, (llcx_i, bot_y), (255, 0, 255),
                           cv2.MARKER_DIAMOND, 16, 3)
            cv2.line(bird_ov, (ccx_i, bot_y), (llcx_i, bot_y), (0, 255, 0), 3)

        overlay_inv   = cv2.warpPerspective(bird_ov, M_inv, ORIGINAL_SIZE,
                                            flags=cv2.INTER_LINEAR)
        overlay_frame = cv2.addWeighted(undist, 1.0, overlay_inv, 0.5, 0)
        overlay_frame = draw_metrics_panel(overlay_frame, self.metrics)

        dbg = build_debug_mosaic_1080p(
            undist, warped, yellow_filt, dash_filt,
            fit_vis, overlay_frame, self.metrics)

        return overlay_frame, dbg


# ═══════════════════════════════════════════════════════════════════════════
#  QoS PROFILES
# ═══════════════════════════════════════════════════════════════════════════

# Images: best-effort depth-1 — never buffer a stale frame
_QOS_IMG = QoSProfile(
    reliability = QoSReliabilityPolicy.RELIABLE,
    history     = QoSHistoryPolicy.KEEP_LAST,
    depth       = 1,
    durability  = QoSDurabilityPolicy.VOLATILE,
)
# Control outputs: reliable, small queue so MPC always has latest
_QOS_CTRL = QoSProfile(
    reliability = QoSReliabilityPolicy.RELIABLE,
    history     = QoSHistoryPolicy.KEEP_LAST,
    depth       = 5,
    durability  = QoSDurabilityPolicy.VOLATILE,
)


# ═══════════════════════════════════════════════════════════════════════════
#  ROS 2 NODE
# ═══════════════════════════════════════════════════════════════════════════

class SumitLaneCVNode(Node):

    def __init__(self):
        super().__init__('sumit_laneCV')

        # ── parameters ───────────────────────────────────────────────────
        self.declare_parameter('mpx',          0.005)
        self.declare_parameter('mpy',          0.005)
        self.declare_parameter('base_frame',  'base_link')
        self.declare_parameter('n_waypoints',  40)

        self._mpx        = self.get_parameter('mpx').value
        self._mpy        = self.get_parameter('mpy').value
        self._base_frame = self.get_parameter('base_frame').value
        self._n_wpts     = self.get_parameter('n_waypoints').value

        self._bridge    = CvBridge()
        self._detector  = LaneDetector()

        # ── video writer (initialised on first frame) ─────────────────────
        self._writer    = None
        self._save_path = DEBUG_VIDEO_SAVE_PATH
        self._video_fps = DEBUG_VIDEO_FPS

        # ── publishers ───────────────────────────────────────────────────
        self._pub_dbg    = self.create_publisher(
            Image,             '/lane/debug_image',           _QOS_IMG)
        self._pub_err    = self.create_publisher(
            Float32,           '/lane/pixel_error',           _QOS_CTRL)
        self._pub_coeffs = self.create_publisher(
            Float64MultiArray, '/lane/polynomial_coeffs',     _QOS_CTRL)
        self._pub_status = self.create_publisher(
            String,            '/lane/status',                _QOS_CTRL)
        self._pub_path   = self.create_publisher(
            Path,              '/lane/left_lane_centre_path', _QOS_IMG)

        # ── subscriber ───────────────────────────────────────────────────
        self.create_subscription(Image, '/img/raw_rgb', self._image_cb, _QOS_IMG)

        self.get_logger().info(
            f"sumit_laneCV ready  |  subscribing: /img/raw_rgb  |  "
            f"mpx={self._mpx} m/px  mpy={self._mpy} m/px  |  "
            f"debug video -> {self._save_path}")

    # ─────────────────────────────────────────────────────────────────────
    def _image_cb(self, msg: Image) -> None:

        # decode ───────────────────────────────────────────────────────────
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge failed: {exc}')
            return

        # run pipeline ─────────────────────────────────────────────────────
        _overlay, dbg = self._detector.process_frame(bgr)
        m             = self._detector.metrics
        stamp         = msg.header.stamp

        # ── /lane/debug_image ─────────────────────────────────────────────
        dbg_msg                 = self._bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
        dbg_msg.header.stamp    = stamp
        dbg_msg.header.frame_id = 'camera'
        self._pub_dbg.publish(dbg_msg)

        # ── save debug frame to video ─────────────────────────────────────
        if self._writer is None:
            os.makedirs(os.path.dirname(os.path.abspath(self._save_path)), exist_ok=True)
            self._writer = cv2.VideoWriter(
                self._save_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self._video_fps,
                (CANVAS_W, CANVAS_H),
            )
            if self._writer.isOpened():
                self.get_logger().info(f'Debug video writer opened: {self._save_path}')
            else:
                self.get_logger().error(f'Failed to open video writer: {self._save_path}')
                self._writer = None
        if self._writer is not None:
            self._writer.write(dbg)

        # ── /lane/pixel_error ─────────────────────────────────────────────
        # pixel_error = CAMERA_CENTER_X - llc_bottom
        # +ve -> vehicle LEFT  of lane centre -> steer RIGHT
        # -ve -> vehicle RIGHT of lane centre -> steer LEFT
        err_msg      = Float32()
        err_msg.data = float(m['pixel_error'])
        self._pub_err.publish(err_msg)

        # ── /lane/polynomial_coeffs ───────────────────────────────────────
        # metrics stores y_A, y_B, d_A, d_B but not the C intercepts.
        # Recover C from each poly's known bottom-row x value:
        #   C = x_bot - A*599^2 - B*599
        y_A = float(m['y_A'])
        y_B = float(m['y_B'])
        y_C = (float(m['y_bot']) - y_A * _BOT_ROW**2 - y_B * _BOT_ROW
               if m['y_bot'] != 0.0 else 0.0)

        # Dash C: recover d_bot via the LLC midpoint identity
        #   llc_bottom = (y_bot + d_bot) / 2  =>  d_bot = 2*llc_bottom - y_bot
        d_A = float(m['d_A'])
        d_B = float(m['d_B'])
        if m['llc_ok'] and m['y_bot'] != 0.0:
            d_bot = 2.0 * float(m['llc_bottom']) - float(m['y_bot'])
            d_C   = d_bot - d_A * _BOT_ROW**2 - d_B * _BOT_ROW
        else:
            d_C = 0.0

        # LLC = average of yellow and dash polys
        llc_A = (y_A + d_A) / 2.0
        llc_B = (y_B + d_B) / 2.0
        llc_C = (float(m['llc_bottom']) - llc_A * _BOT_ROW**2 - llc_B * _BOT_ROW
                 if m['llc_ok'] else 0.0)

        coeffs_msg      = Float64MultiArray()
        coeffs_msg.data = [y_A, y_B, y_C,
                           d_A, d_B, d_C,
                           llc_A, llc_B, llc_C]
        self._pub_coeffs.publish(coeffs_msg)

        # ── /lane/status ──────────────────────────────────────────────────
        status_msg      = String()
        status_msg.data = (
            f"{m['status']} | "
            f"Y:{m['y_mode']}({m['pts_yellow']}pts) | "
            f"D:{m['d_mode']}({m['pts_dash']}pts) | "
            f"Y_rec:{'Y' if m['y_recovery'] else 'n'} "
            f"D_rec:{'Y' if m['d_recovery'] else 'n'} | "
            f"err:{m['pixel_error']:+.1f}px | "
            f"llc_ok:{m['llc_ok']}"
        )
        self._pub_status.publish(status_msg)

        # ── /lane/left_lane_centre_path ───────────────────────────────────
        path                 = Path()
        path.header.stamp    = stamp
        path.header.frame_id = self._base_frame

        if m['llc_ok']:
            # Sample from bottom row (nearest to vehicle) up to ROI_TOP (farthest)
            rows = np.linspace(_BOT_ROW, float(ROI_TOP), self._n_wpts)
            for row in rows:
                col = llc_A * row**2 + llc_B * row + llc_C
                if not (0.0 <= col < UNWARPED_SIZE[0]):
                    continue        # outside valid canvas — skip
                # bird's-eye pixel → base_link metres (ROS REP-105)
                base_x =  (_BOT_ROW - row) * self._mpy   # +X forward
                base_y = -(col - CAMERA_CENTER_X) * self._mpx  # +Y left
                # heading from LLC tangent: d(col)/d(row) = 2A*row + B
                tangent = 2.0 * llc_A * row + llc_B
                heading = float(np.arctan2(-tangent * self._mpx / self._mpy, 1.0))
                pose                    = PoseStamped()
                pose.header.stamp       = stamp
                pose.header.frame_id    = self._base_frame
                pose.pose.position.x    = base_x
                pose.pose.position.y    = base_y
                pose.pose.position.z    = 0.0
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = float(np.sin(heading / 2.0))
                pose.pose.orientation.w = float(np.cos(heading / 2.0))
                path.poses.append(pose)

        self._pub_path.publish(path)

    # ─────────────────────────────────────────────────────────────────────
    def destroy_node(self):
        if self._writer is not None:
            self._writer.release()
            self.get_logger().info(f'Debug video saved: {self._save_path}')
        super().destroy_node()


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = SumitLaneCVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()