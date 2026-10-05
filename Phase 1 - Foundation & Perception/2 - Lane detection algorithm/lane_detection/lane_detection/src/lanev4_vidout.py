#!/usr/bin/env python3
"""
Lane Detection v4 — Drivable-Space Segmentation
================================================
File   : lanev4_vidout.py
Author : Sumit
System : Ubuntu 22.04 | OpenCV 4.x | NumPy

v4 REPLACES the polynomial lane-edge fitting of v3 with direct
DRIVABLE vs NON-DRIVABLE segmentation in bird's-eye view:

    occupied  = yellow tape  OR  large white-island blobs
    drivable  = NOT occupied, restricted to ROI,
                largest connected component touching the
                bottom-center of the BEV (i.e. "reachable from
                the car's current position")

This works at T-junctions / intersections / forks because the
drivable blob can be any shape — it is not constrained to a
fixed-width corridor defined by two polynomials.

Heading is computed from the drivable mask itself: for each row
in a near-field band, take the x-centroid of drivable pixels,
then fit a line through (row, centroid_x). This gives a robust
"which way is the road going right in front of me" direction that
degrades gracefully — no dependency on dashes being visible.

Dash detection (v1 pipeline, mostly unchanged) is retained and
shown in the debug mosaic, and is used as a secondary cross-check
on heading when enough aligned dash points are present.

Exports 3 MP4s (same as v3):
    1. debug_grid.mp4
    2. lane_overlay.mp4
    3. occupancy_grid.mp4

Usage:
    python3 lanev4_vidout.py --input run3.mp4
    python3 lanev4_vidout.py --input run3.mp4 --no-display
"""

import argparse
import sys
import os
import cv2
import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
#  GEOMETRY — shared (unchanged from v3)
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

# ── ROI (bird's-eye px) ───────────────────────────────────────────────────
ROI_TOP   = 220
ROI_LEFT  = 0
ROI_RIGHT = 500

# ── Occupancy grid (unchanged semantics) ─────────────────────────────────
OCC_COLS      = 100
OCC_ROWS      = 120
OCC_DISPLAY_W = 400
OCC_DISPLAY_H = 480
OCC_FREE      = 255
OCC_UNKNOWN   = 128
OCC_OCCUPIED  = 0


# ═══════════════════════════════════════════════════════════════════════════
#  DEBUG MOSAIC LAYOUT  (1920 × 1080 canvas) — unchanged from v3
# ═══════════════════════════════════════════════════════════════════════════

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

BOT_N       = 3
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
#  DEBUG MOSAIC HELPERS (unchanged from v3)
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


def render_occ_roi_only(grid:    np.ndarray,
                        metrics: dict,
                        tile_w:  int,
                        tile_h:  int) -> np.ndarray:
    """
    Render the occupancy grid clipped to the rows that correspond to
    the bird's-eye ROI (rows whose pixel centre >= ROI_TOP).
    """
    px_per_row = UNWARPED_SIZE[1] / OCC_ROWS

    first_roi_row = 0
    for r in range(OCC_ROWS):
        if (r + 0.5) * px_per_row >= ROI_TOP:
            first_roi_row = r
            break

    roi_grid = grid[first_roi_row:, :]

    colour = np.full((*roi_grid.shape, 3), (25, 25, 25), dtype=np.uint8)
    colour[roi_grid == OCC_FREE]     = (255, 255, 255)
    colour[roi_grid == OCC_UNKNOWN]  = (40,  40,  40)
    colour[roi_grid == OCC_OCCUPIED] = (0,   0,  0)

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

    swatch(10,  6, (255, 255, 255), 'Driveable')
    swatch(10, 24, (0,   0,  0), 'Boundary')
    swatch(10, 42, (40,  40,   40), 'Unknown')

    sc = ((0, 210, 0) if metrics['status'].startswith('OK')
          else (0, 90, 255))
    cv2.putText(leg,
                f"{metrics['status']}  "
                f"Heading:{metrics['heading_deg']:+.1f}deg  "
                f"src:{metrics['heading_src']}",
                (140, 14), F, 0.38, sc, 1)
    cv2.putText(leg,
                f"Drivable:{metrics['drivable_pct']:5.1f}%  "
                f"D_pts:{metrics['pts_dash']:3d}  "
                f"ROI rows {first_roi_row}->{OCC_ROWS}",
                (140, 34), F, 0.36, (170, 170, 170), 1)
    cv2.putText(leg,
                f"YellowPts:{metrics['pts_yellow']:5d}  "
                f"WhitePts:{metrics['pts_white']:5d}",
                (140, 52), F, 0.36, (140, 140, 140), 1)

    return np.vstack([disp, leg])


def build_debug_mosaic_1080p(undist:        np.ndarray,
                              warped:        np.ndarray,
                              boundary_filt: np.ndarray,
                              dash_filt:     np.ndarray,
                              seg_vis:       np.ndarray,
                              occ_grid:      np.ndarray,
                              overlay_frame: np.ndarray,
                              metrics:       dict) -> np.ndarray:
    """
    Compose a 1920 × 1080 debug mosaic from 7 panels:

      Top row  (4 tiles): undistorted | bird's-eye+ROI |
                          boundary mask (yellow+white) | dash-filtered
      Bottom row (3 tiles): drivable segmentation+heading | occupancy
                            grid | drivable overlay
    """
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
         f"pts={metrics['n_dash_pts']}"),
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
                  (ROI_RIGHT - 1, UNWARPED_SIZE[1] - 1),
                  (0, 220, 220), 2)
    cv2.putText(warped_roi, 'ROI',
                (ROI_LEFT + 4, ROI_TOP + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)
    _paste(canvas,
           _make_tile(warped_roi, tw, th, "2: Bird's-Eye + ROI"),
           TOP_COL_X[1], TOP_ROW_Y)

    # panel 3 — boundary mask (yellow + white-island), false-colour
    _paste(canvas,
           _make_tile(boundary_filt, tw, th,
                      "3: Boundary Mask (Yellow+White)"),
           TOP_COL_X[2], TOP_ROW_Y)

    # panel 4 — dash mask (white false-colour)
    df = np.zeros((*dash_filt.shape, 3), dtype=np.uint8)
    df[dash_filt > 0] = (200, 200, 200)
    _paste(canvas,
           _make_tile(df, tw, th, "4: Dash Lines Filtered"),
           TOP_COL_X[3], TOP_ROW_Y)

    # ── bottom row ────────────────────────────────────────────────────────

    # panel 5 — drivable segmentation + heading vector
    _paste(canvas,
           _make_tile(seg_vis, bw, bh,
                      "5: Drivable Segmentation + Heading"),
           BOT_COL_X[0], BOT_ROW_Y)

    # panel 6 — occupancy grid (ROI rows only)
    occ_tile = render_occ_roi_only(occ_grid, metrics, bw, bh)
    _paste(canvas,
           _make_tile(occ_tile, bw, bh, "6: Occupancy Grid (ROI only)"),
           BOT_COL_X[1], BOT_ROW_Y)

    # panel 7 — final drivable overlay
    _paste(canvas,
           _make_tile(overlay_frame, bw, bh, "7: Drivable Overlay", metrics),
           BOT_COL_X[2], BOT_ROW_Y)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════
#  SEGMENTATION CONFIG  (v4 — core)
# ═══════════════════════════════════════════════════════════════════════════

# Yellow tape — widened vs v3 to capture the new tape's H~20-24 core,
# validated against sampled BEV pixels (H 5-95pct ~ [20,24,104] with the
# upper tail caused by overexposed/whitish edge pixels).
#
# V_MIN raised from 120 -> 180: at sharp turns the camera can see
# off-track wood paneling/flooring, which has H~16-18 (overlapping the
# tape's hue range) but is much darker (V~110-155) than the tape
# (V~215-235). V>=180 keeps real tape while excluding wood.
Y_HSV_LOW  = np.array([15,  60, 180], dtype=np.uint8)
Y_HSV_HIGH = np.array([32, 255, 255], dtype=np.uint8)

# White islands — near-achromatic and bright. Road (even lit) stays
# below V~150; white islands sit at V~190-235. S stays low for both,
# so V is the primary discriminator.
WHITE_S_MAX = 40
WHITE_V_MIN = 175

# Small/elongated white blobs (dashes, markings) must NOT be treated as
# occupied "islands". A real non-drivable white island is either:
#   (a) huge regardless of position (>= ISLAND_AREA_HUGE), or
#   (b) large AND touches the edge of the ROI (>= ISLAND_AREA_BORDER) —
#       islands always extend to/beyond the visible ROI edge, whereas
#       dashes/markings sit fully inside the road and never reach the
#       ROI border even when their area or extent happens to be large.
ISLAND_AREA_HUGE    = 3000
ISLAND_AREA_BORDER  = 1500
ISLAND_BORDER_MARGIN = 3

# Morphology
BOUND_CLOSE_K = (5, 5)   # close small gaps in yellow/white boundary
BOUND_OPEN_K  = (3, 3)   # remove HSV speckle noise

# ── Heading estimation ────────────────────────────────────────────────────
# Row-wise centroid of the drivable mask in a near-field band, fit with a
# WEIGHTED line (x = a*y + b) — rows closer to the vehicle (larger y) get
# more weight, so a one-sided ROI obstruction far ahead (e.g. an island
# corner cutting into the top of the band) doesn't dominate the fit when
# the road right in front of the vehicle is actually straight.
HEADING_BAND_H   = 140   # px from bottom of BEV used for heading fit
HEADING_ROW_STEP = 2
HEADING_MIN_ROWS = 5
HEADING_MIN_PIX_PER_ROW = 5

# Smoothing / validation
HEADING_SMOOTH_N     = 8
HEADING_MAX_JUMP_DEG = 25.0   # reject single-frame jumps larger than this
HEADING_RECOVERY_THR = 6

# Status thresholds
STATUS_MIN_DRIVABLE_PCT = 5.0   # below this -> NO_DRIVABLE_AREA


# ═══════════════════════════════════════════════════════════════════════════
#  CENTER DASH CONFIG
# ═══════════════════════════════════════════════════════════════════════════
#
# v4 detects dashes via TWO independent channels, unioned together:
#
#   1. EDGE channel (v1, retained) — CLAHE -> adaptive threshold ->
#      Canny -> contour area/aspect/solidity filter. Works well for
#      high-contrast, fully-visible dashes.
#
#   2. BLOB channel (v4, new) — reuses the white-pixel mask from
#      build_boundary_mask(): dash candidates are white blobs that are
#      NOT classified as islands, filtered by area + aspect ratio.
#      Catches dashes the edge channel misses (distant/partial/blurred
#      dashes whose Canny outline is hollow and fails the solidity
#      check) — validated to cut zero-detection frames roughly in
#      half across all three test videos with no false positives on
#      off-track / wood-floor frames (those are excluded because the
#      visible white floor patches touch the ROI border and are
#      classified as islands).

# -- edge channel --
D_CLAHE_CLIP  = 1.5
D_CLAHE_GRID  = (8, 8)
D_ADAPT_BLOCK = 199
D_ADAPT_C     = -45
D_CANNY_LOW   = 215
D_CANNY_HIGH  = 170

D_MIN_AREA     = 120
D_MAX_AREA     = 2000
D_MIN_ASPECT   = 1.4
D_MIN_SOLIDITY = 0.7

_DASH_CLAHE = cv2.createCLAHE(
    clipLimit=D_CLAHE_CLIP, tileGridSize=D_CLAHE_GRID)

# -- blob channel --
DASH_BLOB_AREA_MIN   = 40
DASH_BLOB_AREA_MAX   = 6000
DASH_BLOB_MIN_ASPECT = 1.8

# -- dash point extraction (for future polynomial fit) --
# Applied after the union of edge+blob channels to produce a clean
# list of (cx, cy, ang_vert) centroid points, one per dash instance.
DASH_PT_MAX_ROW_WIDTH   = 18    # px: max horizontal spread at any single row
                                 # handles thin dashes cleanly; merged blobs
                                 # will have many rows wider than this
DASH_PT_MIN_THIN_ROWS   = 5     # minimum rows passing the width check
DASH_PT_MIN_THIN_COV    = 0.60  # fraction of blob height that must be thin;
                                 # rejects island-tarp fragments (ratio≈0.23)
                                 # while keeping motion-blurred dashes (≥0.76)
DASH_PT_MIN_LONG        = 15    # px: minimum long-axis extent (4*std)
DASH_PT_MAX_HEADING_DIFF = 25   # deg: max angle difference from drivable heading
DASH_PT_BOUNDARY_PROX   = 25    # px: dilation radius for boundary proximity check
DASH_PT_ROI_X_MIN       = 20    # px: reject centroids too close to BEV left edge
DASH_PT_ROI_X_MAX       = 480   # px: reject centroids too close to BEV right edge


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — SEGMENTATION  (v4 core)
# ═══════════════════════════════════════════════════════════════════════════

def build_boundary_mask(bgr_warped: np.ndarray):
    """
    Build the "non-drivable boundary" mask = yellow tape OR large
    white-island blobs (dashes / small white markings are excluded).

    Returns
    -------
    boundary_mask : uint8 mask, 255 = occupied/boundary
    yellow_mask   : uint8 mask, 255 = yellow tape pixels
    white_mask    : uint8 mask, 255 = ALL near-white pixels
                    (islands + dashes, before size filtering)
    """
    hsv = cv2.cvtColor(bgr_warped, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, Y_HSV_LOW, Y_HSV_HIGH)
    white_mask  = ((hsv[..., 1] < WHITE_S_MAX) &
                   (hsv[..., 2] > WHITE_V_MIN)).astype(np.uint8) * 255

    # classify white blobs: huge, or large-and-touching-ROI-border -> island
    # (occupied); everything else -> dash/marking (left as drivable)
    white_islands = np.zeros_like(white_mask)
    n, labels, stats_, _ = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8)
    H_img, W_img = white_mask.shape
    m = ISLAND_BORDER_MARGIN
    for lbl in range(1, n):
        area = stats_[lbl, cv2.CC_STAT_AREA]
        x = stats_[lbl, cv2.CC_STAT_LEFT]
        y = stats_[lbl, cv2.CC_STAT_TOP]
        w = stats_[lbl, cv2.CC_STAT_WIDTH]
        h = stats_[lbl, cv2.CC_STAT_HEIGHT]
        touches_border = (x <= m or y <= ROI_TOP + m or
                          x + w >= W_img - m or y + h >= H_img - m)
        is_island = (area >= ISLAND_AREA_HUGE or
                     (area >= ISLAND_AREA_BORDER and touches_border))
        if is_island:
            white_islands[labels == lbl] = 255

    boundary = cv2.bitwise_or(yellow_mask, white_islands)

    # morphological cleanup: open to remove speckle, close to seal gaps
    boundary = cv2.morphologyEx(
        boundary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, BOUND_OPEN_K))
    boundary = cv2.morphologyEx(
        boundary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, BOUND_CLOSE_K))

    return boundary, yellow_mask, white_mask, white_islands


def extract_dash_mask_white(white_mask: np.ndarray,
                             white_islands: np.ndarray) -> np.ndarray:
    """
    Dash candidates = white pixels that are NOT part of a white island
    (see build_boundary_mask), restricted to the ROI and filtered by
    size + aspect ratio.

    This is the v4 "blob" dash channel: dashes are solid filled blobs
    in the same white mask used for island/boundary segmentation, so
    no separate edge/threshold pipeline is needed. It catches dashes
    that the legacy Canny-edge channel (extract_dash_mask_edges) misses
    when the dash's Canny outline is hollow/jagged and fails the
    solidity check (common for distant, partially-occluded, or
    motion-blurred dashes).
    """
    cand = cv2.bitwise_and(white_mask, cv2.bitwise_not(white_islands))
    cand[:ROI_TOP, :] = 0

    out = np.zeros_like(cand)
    n, labels, stats_, _ = cv2.connectedComponentsWithStats(
        cand, connectivity=8)
    for lbl in range(1, n):
        area = stats_[lbl, cv2.CC_STAT_AREA]
        w    = stats_[lbl, cv2.CC_STAT_WIDTH]
        h    = stats_[lbl, cv2.CC_STAT_HEIGHT]
        if area < DASH_BLOB_AREA_MIN or area > DASH_BLOB_AREA_MAX:
            continue
        aspect = max(h, w) / (min(h, w) + 1e-5)
        if aspect < DASH_BLOB_MIN_ASPECT:
            continue
        out[labels == lbl] = 255
    return out


def extract_dash_points(dash_filt: np.ndarray,
                        heading_deg: float,
                        boundary_mask: np.ndarray) -> list:
    """
    Extract a clean list of dash centroid points from the combined
    edge+blob dash mask, ready for downstream polynomial fitting.

    Each returned dict contains:
        cx, cy        : centroid of the dash's thin-row pixels (float,
                        BEV pixel coords). These are the fit input points.
        ang_vert      : long-axis angle relative to vertical (degrees).
                        0 = perfectly straight ahead; positive = leaning
                        right. Use for orientation-consistency checks.
        heading_diff  : |ang_vert - heading_deg| (degrees, 0-90 range).
        long_len      : 4*std along the long axis (approx dash length px).
        n_thin_rows   : number of rows whose horizontal spread ≤ max_row_width.
        thin_coverage : n_thin_rows / total_row_span (quality metric;
                        real dashes ≥ 0.76, tarp fragments ≈ 0.23).
        thin_xs, thin_ys : per-row centroid lists (for refined fitting).

    Filters applied in order:
      1. PCA orientation must agree with heading within DASH_PT_MAX_HEADING_DIFF.
      2. Long-axis extent ≥ DASH_PT_MIN_LONG.
      3. Enough rows are "thin" (≤ DASH_PT_MAX_ROW_WIDTH wide): thin-row
         count ≥ DASH_PT_MIN_THIN_ROWS AND fraction ≥ DASH_PT_MIN_THIN_COV.
         The coverage ratio is the key discriminator against disconnected
         island-tarp fragments, which are wide in most rows but happen to
         have a few thin rows that mislead a pure count check.
      4. Centroid x inside [DASH_PT_ROI_X_MIN, DASH_PT_ROI_X_MAX] (rejects
         fragments at BEV edges introduced by the warp distortion).
      5. Blob pixels must not overlap the boundary_mask when dilated by
         DASH_PT_BOUNDARY_PROX pixels (rejects island-edge artefacts
         directly adjacent to yellow tape).
    """
    kern = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (DASH_PT_BOUNDARY_PROX * 2 + 1, DASH_PT_BOUNDARY_PROX * 2 + 1))
    boundary_dilated = cv2.dilate(boundary_mask, kern)

    n, labels, stats_, centroids = cv2.connectedComponentsWithStats(
        dash_filt, connectivity=8)
    pts = []

    for lbl in range(1, n):
        area = stats_[lbl, cv2.CC_STAT_AREA]
        if area < 20:
            continue

        ys, xs = np.where(labels == lbl)
        coords  = np.column_stack([xs, ys]).astype(np.float64)
        pts_c   = coords - coords.mean(axis=0)
        cov     = np.cov(pts_c.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        principal = eigvecs[:, np.argmax(eigvals)]
        long_len  = 4.0 * np.sqrt(max(eigvals[1], 1e-6))
        ang_vert  = float(np.degrees(
            np.arctan2(principal[0], principal[1])))
        diff = abs(ang_vert - heading_deg)
        diff = float(min(diff, abs(diff - 180.0), abs(diff + 180.0)))

        # filter 1 + 2: orientation and length
        if diff > DASH_PT_MAX_HEADING_DIFF or long_len < DASH_PT_MIN_LONG:
            continue

        # filter 3: thin-row scan and coverage ratio
        thin_xs, thin_ys = [], []
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
        thin_coverage = len(thin_ys) / total_row_span
        if thin_coverage < DASH_PT_MIN_THIN_COV:
            continue

        cx = float(np.mean(thin_xs))
        cy = float(np.mean(thin_ys))

        # filter 4: ROI x-margin
        if cx < DASH_PT_ROI_X_MIN or cx > DASH_PT_ROI_X_MAX:
            continue

        # filter 5: boundary proximity
        comp_mask = (labels == lbl).astype(np.uint8) * 255
        bfrac = (float(np.count_nonzero(
            cv2.bitwise_and(comp_mask, boundary_dilated))) / area)
        if bfrac > 0.3:
            continue

        pts.append({
            'cx':           cx,
            'cy':           cy,
            'ang_vert':     ang_vert,
            'heading_diff': diff,
            'long_len':     long_len,
            'n_thin_rows':  len(thin_ys),
            'thin_coverage': thin_coverage,
            'thin_xs':      thin_xs,
            'thin_ys':      thin_ys,
        })

    # sort by y descending (nearest to vehicle first)
    pts.sort(key=lambda p: -p['cy'])
    return pts


def build_drivable_mask(boundary_mask: np.ndarray):
    """
    drivable = NOT boundary, restricted to ROI, largest connected
    component touching the bottom-center of the BEV.

    Returns
    -------
    drivable     : uint8 mask, 255 = drivable
    seed_label   : connected-component label that was selected
                   (0 if no drivable region found at the seed point)
    n_components : number of candidate drivable components in ROI
    """
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
    else:
        if n > 1:
            areas = stats_[1:, cv2.CC_STAT_AREA]
            seed_label = 1 + int(np.argmax(areas))
            drivable[labels == seed_label] = 255

    n_components = max(0, n - 1)
    return drivable, seed_label, n_components


# ═══════════════════════════════════════════════════════════════════════════
#  HEADING ESTIMATION  (v4 core)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_heading(drivable: np.ndarray):
    """
    Fit x = a*y + b through the row-wise centroid of drivable pixels
    in a near-field band. Returns:

        angle_deg : heading angle in degrees, 0 = straight ahead,
                    positive = road centre drifts to the RIGHT as you
                    look further away (i.e. car should steer right)
        rows, centroids : the raw points used (for visualisation)
        valid : bool, whether enough points were found
    """
    rows = []
    centroids = []
    y_bottom = UNWARPED_SIZE[1] - 1
    y_top    = max(ROI_TOP, UNWARPED_SIZE[1] - HEADING_BAND_H)

    for y in range(y_bottom, y_top - 1, -HEADING_ROW_STEP):
        xs = np.where(drivable[y, :] > 0)[0]
        if len(xs) >= HEADING_MIN_PIX_PER_ROW:
            rows.append(y)
            centroids.append(float(xs.mean()))

    if len(rows) < HEADING_MIN_ROWS:
        return 0.0, np.array(rows), np.array(centroids), False

    rows_arr = np.array(rows, dtype=np.float64)
    cent_arr = np.array(centroids, dtype=np.float64)

    # weight rows closer to the vehicle (larger y) more heavily, so a
    # one-sided obstruction near the far edge of the band doesn't
    # dominate the fit when the near-field road is straight.
    y_max = rows_arr.max()
    y_min = rows_arr.min()
    span  = max(y_max - y_min, 1.0)
    weights = 0.2 + 0.8 * (rows_arr - y_min) / span

    a, b = np.polyfit(rows_arr, cent_arr, 1, w=weights)
    angle_deg = float(np.degrees(np.arctan(a)))

    return angle_deg, rows_arr, cent_arr, True


def estimate_dash_heading(dash_mask: np.ndarray):
    """
    Secondary heading cross-check from dash points within the ROI,
    using the same row-wise-centroid-line-fit approach restricted to
    dash pixels. Returns (angle_deg|None, n_pts).
    """
    rows = []
    centroids = []
    y_bottom = UNWARPED_SIZE[1] - 1
    y_top    = max(ROI_TOP, UNWARPED_SIZE[1] - HEADING_BAND_H)

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
    a, b = np.polyfit(rows_arr, cent_arr, 1)
    return float(np.degrees(np.arctan(a))), n_pts


# ═══════════════════════════════════════════════════════════════════════════
#  CV HELPERS — CENTER DASHES  (v1 logic, retained from v3)
# ═══════════════════════════════════════════════════════════════════════════

def dash_preprocess_edges(gray: np.ndarray):
    """
    EDGE channel preprocessing (v1, retained): CLAHE -> adaptive
    threshold -> morphological open -> GaussianBlur -> Canny -> AND
    with tape mask.
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


def extract_dash_mask_edges(roi_mask: np.ndarray) -> np.ndarray:
    """
    EDGE channel contour filter (v1, retained): morphological close +
    dilate, then area / aspect / solidity checks.
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


# ═══════════════════════════════════════════════════════════════════════════
#  OCCUPANCY GRID  (v4 — derived directly from segmentation)
# ═══════════════════════════════════════════════════════════════════════════

def build_occupancy_grid(drivable: np.ndarray,
                          boundary: np.ndarray) -> np.ndarray:
    """
    Build a (OCC_ROWS x OCC_COLS) occupancy grid directly from the
    per-pixel segmentation. Cells outside the ROI are UNKNOWN.
    A cell is OCCUPIED if its majority of pixels are boundary pixels,
    FREE if its majority of pixels are drivable, else UNKNOWN.
    """
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
            n_occ  = int(np.count_nonzero(cell_boundary))

            if n_occ / n_total > 0.15:
                grid[row_idx, col_idx] = OCC_OCCUPIED
            elif n_free / n_total > 0.5:
                grid[row_idx, col_idx] = OCC_FREE
            else:
                grid[row_idx, col_idx] = OCC_UNKNOWN

    return grid


def render_occupancy_grid(grid: np.ndarray, metrics: dict) -> np.ndarray:
    """
    Full-size standalone occupancy grid image for the occupancy_grid.mp4
    export.
    """
    colour = np.zeros((OCC_ROWS, OCC_COLS, 3), dtype=np.uint8)
    colour[grid == OCC_FREE]     = (255, 255, 255)
    colour[grid == OCC_UNKNOWN]  = (30,  30,  30)
    colour[grid == OCC_OCCUPIED] = (0,   0,   0)

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
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sc, 1)
    cv2.putText(legend,
                f"Drivable: {metrics['drivable_pct']:.1f}%   "
                f"Heading: {metrics['heading_deg']:+.1f}deg "
                f"({metrics['heading_src']})",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(legend,
                f"Yellow pts: {metrics['pts_yellow']:5d}   "
                f"White pts: {metrics['pts_white']:5d}   "
                f"Dash pts: {metrics['pts_dash']:3d}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(legend,
                f"Grid {OCC_COLS}x{OCC_ROWS}  "
                f"({UNWARPED_SIZE[0]/OCC_COLS:.1f}x"
                f"{UNWARPED_SIZE[1]/OCC_ROWS:.1f} px/cell)",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

    title = np.zeros((30, OCC_DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(title, 'OCCUPANCY GRID  |  v4 Drivable-Space Pipeline',
                (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    return np.vstack([title, disp, legend])


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def draw_metrics_panel(img: np.ndarray, m: dict) -> np.ndarray:
    panel   = img.copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (560, 340), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)

    C_OK   = (0, 255, 0)
    C_WARN = (0, 165, 255)
    C_BAD  = (0, 0, 255)
    C_INFO = (255, 255, 255)
    C_BND  = (0, 220, 220)
    C_DASH = (200, 200, 100)

    def put(t, y, c=C_INFO):
        cv2.putText(panel, t, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, c, 1)

    sc = (C_OK   if m['status'].startswith('OK')  else
          C_WARN if m['status'].startswith('PARTIAL') else C_BAD)

    put(f"Status       : {m['status']}",        26, sc)
    put(f"Frame        : {m.get('frame', 0):6d}", 50, C_INFO)

    put("-- DRIVABLE SEGMENTATION ----------------", 78, C_BND)
    put(f"  drivable area : {m['drivable_pct']:5.1f}%   "
        f"components : {m['n_components']:2d}",
        100,
        C_OK if m['drivable_pct'] >= STATUS_MIN_DRIVABLE_PCT else C_BAD)
    put(f"  yellow pts : {m['pts_yellow']:5d}   "
        f"white pts : {m['pts_white']:5d}", 122, C_BND)

    put("-- HEADING ------------------------------", 150, C_INFO)
    put(f"  angle : {m['heading_deg']:+6.1f} deg   "
        f"source : {m['heading_src']}   "
        f"rows : {m['heading_rows']:3d}",
        172,
        C_OK if m['heading_src'] != 'NONE' else C_BAD)
    put(f"  raw_angle : {m['heading_raw_deg']:+6.1f} deg   "
        f"jump : {m['heading_jump']:5.1f} (max {HEADING_MAX_JUMP_DEG})  "
        f"recovery : {'YES' if m['heading_recovery'] else 'no'}",
        194,
        C_OK if m['heading_jump'] < HEADING_MAX_JUMP_DEG else C_WARN)

    dash_ang = m['dash_heading_deg']
    dash_ang_str = 'n/a' if dash_ang is None else f"{dash_ang:+.1f}deg"
    put("-- DASH (secondary heading cross-check) --", 222, C_DASH)
    put(f"  mode:{m['d_mode']}  raw:{m['pts_dash']:4d}px  "
        f"blob:{m['pts_dash_blob']:3d}  edge:{m['pts_dash_edge']:3d}  "
        f"clean_pts:{m['n_dash_pts']:2d}  dash_angle:{dash_ang_str}",
        244, C_DASH)

    put("-- CONFIG --------------------------------", 272, C_INFO)
    put(f"  Y_HSV=[{Y_HSV_LOW[0]}-{Y_HSV_HIGH[0]}]  "
        f"WHITE S<{WHITE_S_MAX} V>{WHITE_V_MIN}",
        294, C_INFO)
    put(f"  DASH_BLOB=[{DASH_BLOB_AREA_MIN},{DASH_BLOB_AREA_MAX}] "
        f"asp>{DASH_BLOB_MIN_ASPECT}  "
        f"HEADING_BAND={HEADING_BAND_H}px smooth={HEADING_SMOOTH_N}",
        316, C_DASH)

    return panel


def build_seg_vis(boundary: np.ndarray,
                  drivable: np.ndarray,
                  rows:     np.ndarray,
                  centroids: np.ndarray,
                  heading_deg: float,
                  heading_valid: bool,
                  dash_points: list = None) -> np.ndarray:
    """
    Bird's-eye visualisation of the drivable segmentation:
    drivable -> green, boundary -> red, unknown -> dark gray.
    Overlay the row-wise centroid points, the fitted heading line,
    and accepted dash centroid points (magenta crosses + thin-row trails).
    """
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

    # draw accepted dash points: thin-row trail (dots) + centroid cross
    if dash_points:
        for dp in dash_points:
            # per-row centroid trail in magenta
            for tx, ty in zip(dp['thin_xs'], dp['thin_ys']):
                cv2.circle(vis, (int(tx), int(ty)), 1, (255, 0, 255), -1)
            # centroid marker: larger cross in bright magenta
            cx_i, cy_i = int(dp['cx']), int(dp['cy'])
            cv2.drawMarker(vis, (cx_i, cy_i), (255, 80, 255),
                           cv2.MARKER_CROSS, 14, 2)

    cv2.putText(vis, 'Green=driv  Red=bnd  Yellow=hdg_cent  '
                     'Magenta=dash_pts  White=heading',
                (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
    return vis


def build_boundary_vis(boundary_mask: np.ndarray,
                        yellow_mask: np.ndarray,
                        white_mask: np.ndarray) -> np.ndarray:
    """False-colour visualisation of the boundary mask for panel 3."""
    vis = np.zeros((*boundary_mask.shape, 3), dtype=np.uint8)
    vis[white_mask > 0]  = (120, 120, 120)   # all white pixels: gray
    vis[yellow_mask > 0] = (0, 215, 255)     # yellow tape: gold
    vis[boundary_mask > 0] = np.where(
        (vis[boundary_mask > 0] == 0).all(axis=-1, keepdims=True),
        (0, 0, 200), vis[boundary_mask > 0])
    # ensure boundary (post-cleanup) regions are visibly highlighted
    highlight = np.zeros_like(vis)
    highlight[boundary_mask > 0] = (0, 0, 200)
    vis = cv2.addWeighted(vis, 0.5, highlight, 0.5, 0)
    return vis


# ═══════════════════════════════════════════════════════════════════════════
#  STATEFUL LANE DETECTOR  (v4)
# ═══════════════════════════════════════════════════════════════════════════

class LaneDetector:
    """
    v4 drivable-space pipeline.

    process_frame() returns the same 4-tuple as v3:
        overlay_frame, dbg, occ_display, occ_grid
    so existing main()/video-export code is unchanged.
    """

    def __init__(self):
        # heading smoothing/validation state
        self._heading_history  = deque(maxlen=HEADING_SMOOTH_N)
        self._heading_rejects  = 0
        self._heading_recovery = False

        self._frame_idx = 0

        self.metrics = {
            'status':           'INIT',
            'frame':            0,
            'drivable_pct':     0.0,
            'n_components':     0,
            'pts_yellow':       0,
            'pts_white':        0,
            'pts_dash':         0,
            'heading_deg':      0.0,
            'heading_raw_deg':  0.0,
            'heading_src':      'NONE',
            'heading_rows':     0,
            'heading_jump':     0.0,
            'heading_recovery': False,
            'dash_heading_deg': None,
            'd_mode':           'INIT',
            'pts_dash_blob':    0,
            'pts_dash_edge':    0,
            'dash_points':      [],
            'n_dash_pts':       0,
        }

    # ─────────────────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray):
        """
        Run the full v4 detection pipeline on one BGR frame.

        Returns
        -------
        overlay_frame : np.ndarray  (ORIGINAL_SIZE BGR)
        dbg           : np.ndarray  (1920x1080 BGR)
        occ_display   : np.ndarray  (standalone occupancy image)
        occ_grid      : np.ndarray  (OCC_ROWS x OCC_COLS uint8)
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
        #  SEGMENTATION  (v4 core)
        # ══════════════════════════════════════════════════════════════════
        boundary_mask, yellow_mask, white_mask, white_islands = \
            build_boundary_mask(warped)

        drivable, seed_label, n_components = build_drivable_mask(
            boundary_mask)

        roi_area = (UNWARPED_SIZE[1] - ROI_TOP) * (ROI_RIGHT - ROI_LEFT)
        drivable_pct = 100.0 * float(np.count_nonzero(drivable)) / \
            max(roi_area, 1)

        pts_yellow = int(np.count_nonzero(yellow_mask[ROI_TOP:, :]))
        pts_white  = int(np.count_nonzero(white_mask[ROI_TOP:, :]))

        # ══════════════════════════════════════════════════════════════════
        #  DASH DETECTION — two channels, unioned
        # ══════════════════════════════════════════════════════════════════
        # channel 1: BLOB (v4) — reuses white_mask/white_islands from the
        # segmentation step above, no extra masking work needed.
        dash_blob = extract_dash_mask_white(white_mask, white_islands)

        # channel 2: EDGE (v1, retained) — CLAHE/adaptive-threshold/Canny
        _, edges_gated, _ = dash_preprocess_edges(gray)
        dash_roi = edges_gated.copy()
        dash_roi[:ROI_TOP, :]   = 0
        dash_roi[:, :ROI_LEFT]  = 0
        dash_roi[:, ROI_RIGHT:] = 0
        dash_edge = extract_dash_mask_edges(dash_roi)

        dash_filt = cv2.bitwise_or(dash_blob, dash_edge)
        pts_dash  = int(np.count_nonzero(dash_filt))
        pts_dash_blob = int(np.count_nonzero(dash_blob))
        pts_dash_edge = int(np.count_nonzero(dash_edge))
        d_mode    = 'DETECT' if pts_dash > 0 else 'LOW_PTS'

        # ══════════════════════════════════════════════════════════════════
        #  HEADING  (v4 core) — primary: drivable-mask row centroid fit
        # ══════════════════════════════════════════════════════════════════
        raw_angle, h_rows, h_centroids, h_valid = estimate_heading(drivable)

        heading_jump = 0.0
        if h_valid:
            if len(self._heading_history) > 0:
                last = float(np.mean(self._heading_history))
                heading_jump = abs(raw_angle - last)

                if heading_jump <= HEADING_MAX_JUMP_DEG or \
                        self._heading_recovery:
                    self._heading_history.append(raw_angle)
                    self._heading_rejects  = 0
                    self._heading_recovery = False
                else:
                    self._heading_rejects += 1
                    if self._heading_rejects >= HEADING_RECOVERY_THR:
                        self._heading_recovery = True
                        self._heading_history.clear()
                        self._heading_history.append(raw_angle)
            else:
                self._heading_history.append(raw_angle)
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

        # ── secondary: dash-based cross-check (reported only) ─────────────
        # extract_dash_points needs the final smoothed heading so it can
        # filter by orientation — compute it after the history update.
        dash_points = extract_dash_points(dash_filt, heading_deg, boundary_mask)
        dash_heading_deg, _ = estimate_dash_heading(dash_filt)

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
        self.metrics.update({
            'status':           status,
            'frame':            self._frame_idx,
            'drivable_pct':     drivable_pct,
            'n_components':     n_components,
            'pts_yellow':       pts_yellow,
            'pts_white':        pts_white,
            'pts_dash':         pts_dash,
            'heading_deg':      heading_deg,
            'heading_raw_deg':  raw_angle,
            'heading_src':      heading_src,
            'heading_rows':     int(len(h_rows)),
            'heading_jump':     heading_jump,
            'heading_recovery': self._heading_recovery,
            'dash_heading_deg': dash_heading_deg,
            'd_mode':           d_mode,
            'pts_dash_blob':    pts_dash_blob,
            'pts_dash_edge':    pts_dash_edge,
            'dash_points':      dash_points,
            'n_dash_pts':       len(dash_points),
        })

        # ══════════════════════════════════════════════════════════════════
        #  OUTPUTS
        # ══════════════════════════════════════════════════════════════════
        occ_display = render_occupancy_grid(occ_grid, self.metrics)

        seg_vis = build_seg_vis(boundary_mask, drivable,
                                 h_rows, h_centroids,
                                 heading_deg, h_valid or
                                 bool(self._heading_history),
                                 dash_points=dash_points)

        boundary_vis = build_boundary_vis(boundary_mask, yellow_mask,
                                           white_mask)

        # ── drivable overlay → unwarp back to camera view ──────────────────
        bird_ov = np.zeros(
            (UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)
        bird_ov[drivable > 0] = (0, 180, 0)
        bird_ov[boundary_mask > 0] = (0, 0, 200)

        # heading arrow on the bird's-eye overlay
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
        overlay_frame = draw_metrics_panel(overlay_frame, self.metrics)

        dbg = build_debug_mosaic_1080p(
            undist        = undist,
            warped        = warped,
            boundary_filt = boundary_vis,
            dash_filt     = dash_filt,
            seg_vis       = seg_vis,
            occ_grid      = occ_grid,
            overlay_frame = overlay_frame,
            metrics       = self.metrics
        )

        return overlay_frame, dbg, occ_display, occ_grid


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEO I/O  (unchanged from v3)
# ═══════════════════════════════════════════════════════════════════════════

def open_writer(path: str, w: int, h: int, fps: float):
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer


def progress_bar(cur: int, total: int, bar_len: int = 50) -> str:
    filled = int(bar_len * cur / max(total, 1))
    bar    = '#' * filled + '.' * (bar_len - filled)
    return (f"\r[{bar}] {cur:6d}/{total:6d}  "
            f"({100 * cur / max(total, 1):5.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Lane Detection v4 — drivable-space video inference')
    p.add_argument('--input',       '-i', required=True)
    p.add_argument('--out-debug',   default='v4_debug_grid.mp4')
    p.add_argument('--out-overlay', default='v4_lane_overlay.mp4')
    p.add_argument('--out-occ',     default='v4_occupancy_grid.mp4')
    p.add_argument('--no-display',  action='store_true')
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

    print(f"\n{'-'*62}")
    print(f"  Input  : {args.input}  "
          f"[{src_w}x{src_h}  {fps:.2f}fps  {total}fr]")
    print(f"{'-'*62}\n")

    start    = max(0, args.start_frame)
    end      = total if args.end_frame < 0 else min(args.end_frame, total)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    n_frames = end - start

    detector = LaneDetector()

    # ── probe output sizes from the first frame ───────────────────────────
    ret, first = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame.")
        sys.exit(1)

    ov_f, dbg_f, occ_f, _ = detector.process_frame(first.copy())
    ov_h,  ov_w  = ov_f.shape[:2]
    dbg_h, dbg_w = dbg_f.shape[:2]
    occ_h, occ_w = occ_f.shape[:2]

    assert dbg_w == CANVAS_W and dbg_h == CANVAS_H, \
        f"Unexpected mosaic size {dbg_w}x{dbg_h}"

    try:
        w_ov  = open_writer(args.out_overlay, ov_w,      ov_h,      fps)
        w_dbg = open_writer(args.out_debug,   CANVAS_W,  CANVAS_H,  fps)
        w_occ = open_writer(args.out_occ,     occ_w,     occ_h,     fps)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        cap.release()
        sys.exit(1)

    print(f"  overlay   -> {args.out_overlay}  [{ov_w}x{ov_h}]")
    print(f"  debug     -> {args.out_debug}    [{CANVAS_W}x{CANVAS_H}]")
    print(f"  occupancy -> {args.out_occ}      [{occ_w}x{occ_h}]\n")

    w_ov.write(ov_f)
    w_dbg.write(dbg_f)
    w_occ.write(occ_f)

    SHOW = not args.no_display
    if SHOW:
        PREV_W = 1280
        PREV_H = int(dbg_h * PREV_W / max(dbg_w, 1))
        cv2.namedWindow('Debug',      cv2.WINDOW_NORMAL)
        cv2.namedWindow('Overlay',    cv2.WINDOW_NORMAL)
        cv2.namedWindow('Occupancy',  cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Debug',     PREV_W, PREV_H)
        cv2.resizeWindow('Overlay',   1280, 720)
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
        w_ov.write(ov_f)
        w_dbg.write(dbg_f)
        w_occ.write(occ_f)

        print(progress_bar(fn, n_frames), end='', flush=True)

        if SHOW:
            cv2.imshow('Debug',     dbg_f)
            cv2.imshow('Overlay',   cv2.resize(ov_f, (1280, 720)))
            cv2.imshow('Occupancy', occ_f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Quit.")
                break

    print(f"\n\n{'-'*62}")
    print(f"  Done.  {n_frames} frames processed.")
    print(f"  {args.out_overlay}")
    print(f"  {args.out_debug}")
    print(f"  {args.out_occ}")
    print(f"{'-'*62}\n")

    cap.release()
    w_ov.release()
    w_dbg.release()
    w_occ.release()

    if SHOW:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
