#!/usr/bin/env python3
"""
Node name  : sumit_laneCV
File name  : sumit_laneCV.py
Author     : Sumit
System     : Ubuntu 22.04 | JetPack 6.2.2 | ROS2 Humble
             DepthAI 3.6.1 API v3 | OpenCV 4.13.0 | NumPy 1.23.5
Description: CV-based lane detection on OAK camera live feed.
             Uses DepthAI v3 API (no XLinkOut nodes).
             Publishes annotated image, debug grid, occupancy grid,
             lane coefficients and status string.
"""

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

import sys

print("EXECUTABLE:", sys.executable)
print("OPENCV:", cv2.__version__)
print("DEPTHAI:", dai.__version__)

from collections import deque


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS / CONFIG
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

CANNY_LOW  = 255
CANNY_HIGH = 255

CLAHE_CLIP = 1.5
CLAHE_GRID = (8, 8)

ADAPT_BLOCK = 199
ADAPT_C     = -45

ROI_TOP   = 280
ROI_LEFT  = 50
ROI_RIGHT = 485

MIN_DASH_AREA     = 120
MAX_DASH_AREA     = 2000
MIN_DASH_ASPECT   = 1.4
MIN_DASH_SOLIDITY = 0.7

SEARCH_WINDOW          = 100
SEARCH_WINDOW_RECOVERY = 400
MAX_LATERAL_JUMP       = 150
MAX_CURV_JUMP          = 0.15
SMOOTH_N               = 5
RECOVERY_THRESHOLD     = 8
LANE_OFFSET            = 230

OCC_COLS      = 100
OCC_ROWS      = 120
OCC_DISPLAY_W = 400
OCC_DISPLAY_H = 480

OCC_FREE     = 255
OCC_UNKNOWN  = 128
OCC_OCCUPIED = 0

DEBUG_W = 400
DEBUG_H = 300

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
CLAHE = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


# ═══════════════════════════════════════════════════════════════════════════
#  PURE CV HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_lane(gray: np.ndarray):
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
    close_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed   = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, close_k)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated  = cv2.dilate(closed, dilate_k, iterations=2)
    filtered = np.zeros_like(dilated)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
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


def is_valid_fit(new_coeffs: np.ndarray,
                 history: deque):
    if len(history) == 0:
        return True, 0.0, 0.0
    last_coeffs = np.mean(history, axis=0)
    lat_jump    = abs(np.polyval(new_coeffs,  600)
                    - np.polyval(last_coeffs, 600))
    curv_jump   = abs(new_coeffs[0] - last_coeffs[0])
    valid = (lat_jump <= MAX_LATERAL_JUMP) and (curv_jump <= MAX_CURV_JUMP)
    return valid, float(lat_jump), float(curv_jump)


def build_occupancy_grid(coeffs,
                         left_x:  np.ndarray,
                         right_x: np.ndarray) -> np.ndarray:
    grid       = np.full((OCC_ROWS, OCC_COLS), OCC_UNKNOWN, dtype=np.uint8)
    px_per_col = UNWARPED_SIZE[0] / OCC_COLS
    px_per_row = UNWARPED_SIZE[1] / OCC_ROWS

    if coeffs is None:
        return grid

    for row_idx in range(OCC_ROWS):
        bird_y = (row_idx + 0.5) * px_per_row
        if bird_y < ROI_TOP:
            continue
        cx = float(np.polyval(coeffs, bird_y))
        lx = cx - LANE_OFFSET
        rx = cx + LANE_OFFSET

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


def render_occupancy_grid(grid: np.ndarray,
                          metrics: dict) -> np.ndarray:
    h, w   = grid.shape
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    colour[grid == OCC_FREE]     = (255, 255, 255)
    colour[grid == OCC_UNKNOWN]  = (0,   0,   0)
    colour[grid == OCC_OCCUPIED] = (0,   0,   0)

    disp = cv2.resize(colour, (OCC_DISPLAY_W, OCC_DISPLAY_H),
                      interpolation=cv2.INTER_NEAREST)

    cell_w     = OCC_DISPLAY_W / OCC_COLS
    cell_h     = OCC_DISPLAY_H / OCC_ROWS
    grid_color = (50, 50, 50)
    for c in range(OCC_COLS + 1):
        x = int(c * cell_w)
        cv2.line(disp, (x, 0), (x, OCC_DISPLAY_H), grid_color, 1)
    for r in range(OCC_ROWS + 1):
        y = int(r * cell_h)
        cv2.line(disp, (0, y), (OCC_DISPLAY_W, y), grid_color, 1)

    legend = np.zeros((110, OCC_DISPLAY_W, 3), dtype=np.uint8)

    def swatch(x: int, y: int, color: tuple, label: str):
        cv2.rectangle(legend, (x, y), (x + 18, y + 14), color, -1)
        if color == (255, 255, 255):
            cv2.rectangle(legend, (x, y), (x + 18, y + 14),
                          (100, 100, 100), 1)
        cv2.putText(legend, label, (x + 22, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    swatch(10,  8, (255, 255, 255), 'Free / driveable')
    swatch(10, 30, (0,   0,   0),   'Lane boundary / obstacle')
    swatch(10, 52, (0,   0,   0),   'Unknown / outside ROI')

    sc = ((0, 220, 0)
          if metrics['status'].startswith('OK')
          else (0, 100, 255))
    cv2.putText(legend,
                f"Status: {metrics['status']}  "
                f"Recover: {'YES' if metrics['recovery'] else 'no'}",
                (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sc, 1)
    cv2.putText(legend,
                f"Grid {OCC_COLS}x{OCC_ROWS}  "
                f"({UNWARPED_SIZE[0]/OCC_COLS:.1f}x"
                f"{UNWARPED_SIZE[1]/OCC_ROWS:.1f} px/cell)",
                (10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (180, 180, 180), 1)

    title = np.zeros((28, OCC_DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(title, 'OCCUPANCY GRID',
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 255, 255), 1)

    return np.vstack([title, disp, legend])


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 11 + 10, 28), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return out


def to_tile(img: np.ndarray,
            w: int = DEBUG_W,
            h: int = DEBUG_H) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (w, h))


def draw_metrics(img: np.ndarray, m: dict) -> np.ndarray:
    panel   = img.copy()
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (440, 290), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)

    C_OK   = (0, 255, 0)
    C_WARN = (0, 165, 255)
    C_BAD  = (0, 0, 255)
    C_INFO = (255, 255, 255)

    def put(text: str, y: int, color: tuple = C_INFO):
        cv2.putText(panel, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)

    sc = (C_OK   if m['status'].startswith('OK')  else
          C_WARN if m['status'].startswith('REJ') else C_BAD)

    put(f"Status   : {m['status']}",  30,  sc)
    put(f"Recovery : {'YES' if m['recovery'] else 'no'}", 58,
        C_WARN if m['recovery'] else C_OK)
    put(f"Points   : {m['pts']}", 86,
        C_OK if m['pts'] >= 20 else C_BAD)
    put(f"Center X : {m['center_x']}", 114)
    put(f"A (curv) : {m['A']:+.6f}",  142)
    put(f"B (slope): {m['B']:+.4f}",  170)
    put(f"C (pos)  : {m['C']:+.2f}",  198)
    put(f"Lat jump : {m['lateral_jump']:6.2f} px  (max {MAX_LATERAL_JUMP})",
        226, C_OK if m['lateral_jump'] < MAX_LATERAL_JUMP else C_BAD)
    put(f"Curv jump: {m['curv_jump']:.5f}   (max {MAX_CURV_JUMP})",
        254, C_OK if m['curv_jump'] < MAX_CURV_JUMP else C_BAD)
    put(f"CLAHE={CLAHE_CLIP}  Canny={CANNY_LOW}/{CANNY_HIGH}  "
        f"AdaptB={ADAPT_BLOCK}", 282)
    return panel


def build_debug_grid(undist:      np.ndarray,
                     warped:      np.ndarray,
                     enhanced:    np.ndarray,
                     tape_mask:   np.ndarray,
                     edges_gated: np.ndarray,
                     filtered:    np.ndarray,
                     fit_vis:     np.ndarray,
                     occ_display: np.ndarray) -> np.ndarray:
    t1 = add_label(to_tile(undist),      "1: Undistorted")
    t2 = add_label(to_tile(warped),      "2: Warped")
    t3 = add_label(to_tile(enhanced),    "3: CLAHE")
    t4 = add_label(to_tile(tape_mask),   "4: Tape Mask")
    t5 = add_label(to_tile(edges_gated), "5: Gated Edges")
    t6 = add_label(to_tile(filtered),    "6: Dash Filter")
    t7 = add_label(to_tile(fit_vis),     "7: Fit")
    t8 = add_label(
        cv2.resize(occ_display, (DEBUG_W, DEBUG_H)),
        "8: Occupancy"
    )
    return np.vstack([
        np.hstack([t1, t2, t3, t4]),
        np.hstack([t5, t6, t7, t8])
    ])


# ═══════════════════════════════════════════════════════════════════════════
#  ROS 2 NODE
# ═══════════════════════════════════════════════════════════════════════════

class SumitLaneCV(Node):
    """
    ROS 2 Humble node — DepthAI v3 API (no XLinkOut nodes).
    Camera  : OAK via dai.node.Camera + requestOutput()
    CV      : OpenCV 4.13.0
    Publishes:
        /lane/result_image      sensor_msgs/Image
        /lane/debug_grid        sensor_msgs/Image
        /lane/occupancy_image   sensor_msgs/Image
        /lane/occupancy_grid    nav_msgs/OccupancyGrid
        /lane/coefficients      std_msgs/Float32MultiArray  [A, B, C]
        /lane/status            std_msgs/String  (JSON)
    """

    def __init__(self):
        super().__init__('sumit_laneCV')

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter('camera_fps',   45)
        self.declare_parameter('show_windows', False)

        self.cam_fps  = int(self.get_parameter('camera_fps').value)
        self.show_win = bool(self.get_parameter('show_windows').value)

        # ── QoS ───────────────────────────────────────────────────────────
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_result  = self.create_publisher(
            Image,             '/lane/result_image',    qos_best_effort)
        self.pub_debug   = self.create_publisher(
            Image,             '/lane/debug_grid',      qos_best_effort)
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

        # ── Algorithm state ───────────────────────────────────────────────
        self._coeff_history       = deque(maxlen=SMOOTH_N)
        self._last_center_x       = None
        self._consecutive_rejects = 0
        self._recovery_mode       = False
        self._metrics = {
            'status': 'INIT', 'pts': 0,   'center_x': 0,
            'lateral_jump': 0.0,           'curv_jump': 0.0,
            'A': 0.0, 'B': 0.0, 'C': 0.0,
            'smooth_n': 0,                 'recovery': False,
        }

        # ── DepthAI v3 pipeline ───────────────────────────────────────────
        self._pipeline     = None
        self._video_queue  = None
        self._init_depthai()

        # ── Timer ─────────────────────────────────────────────────────────
        self._timer = self.create_timer(0.01, self._timer_cb)

        self.get_logger().info(
            f'sumit_laneCV ready  |  '
            f'DepthAI {dai.__version__} (v3 API)  |  '
            f'OpenCV {cv2.__version__}'
        )

    # ──────────────────────────────────────────────────────────────────────
    #  DepthAI v3 API — NO XLinkOut nodes
    # ──────────────────────────────────────────────────────────────────────

    def _init_depthai(self):
        """
        DepthAI 3.6.1 v3 API — official pattern:
            1. pipeline.create(dai.node.Camera).build(...)
            2. cam.requestOutput((w,h), type=...)
            3. output.createOutputQueue()
            4. pipeline.start()
        NO XLinkOut nodes required.
        """
        self._pipeline = dai.Pipeline()

        # ── Camera node ───────────────────────────────────────────────────
        cam = self._pipeline.create(dai.node.Camera).build(
            boardSocket=dai.CameraBoardSocket.CAM_A,
            sensorResolution=(ORIGINAL_SIZE[0], ORIGINAL_SIZE[1]),
            sensorFps=float(self.cam_fps)
        )

        # ── Request output ────────────────────────────────────────────────
        # requestOutput auto-creates the output stream (no XLinkOut needed)
        video_out = cam.requestOutput(
            size=(ORIGINAL_SIZE[0], ORIGINAL_SIZE[1]),
            type=dai.ImgFrame.Type.BGR888p
        )

        # ── Image Manipulation Node (180° Rotation) ───────────────────────
        manip = self._pipeline.create(dai.node.ImageManip)
        manip.initialConfig.addRotateDeg(180)
        
        # FIX: Explicitly allocate enough buffer pool memory for the 1280x720 BGR frame
        # Width * Height * 3 channels
        max_bytes = ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1] * 3
        manip.setMaxOutputFrameSize(max_bytes)
        
        # Link the camera stream into the image manipulation hardware block
        video_out.link(manip.inputImage)

        # ── Create output queue (v3 API) ──────────────────────────────────
        self._video_queue = manip.out.createOutputQueue(
            maxSize=1,
            blocking=False
        )

        # ── Start pipeline ────────────────────────────────────────────────
        self._pipeline.start()

        self.get_logger().info(
            f'DepthAI OAK camera started  '
            f'({ORIGINAL_SIZE[0]}x{ORIGINAL_SIZE[1]} '
            f'@ {self.cam_fps}fps)'
        )

    # ──────────────────────────────────────────────────────────────────────
    #  Timer callback
    # ──────────────────────────────────────────────────────────────────────

    def _timer_cb(self):
        if not self._pipeline.isRunning():
            return

        in_rgb = self._video_queue.get() if self._video_queue.has() else None
        if in_rgb is None:
            return

        frame = in_rgb.getCvFrame()

        result, dbg, occ_display, occ_grid = self._detect_lane(frame)

        stamp = self.get_clock().now().to_msg()

        # ── Images ────────────────────────────────────────────────────────
        self.pub_result.publish(
            self.bridge.cv2_to_imgmsg(result,      encoding='bgr8'))
        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(dbg,         encoding='bgr8'))
        self.pub_occ_img.publish(
            self.bridge.cv2_to_imgmsg(occ_display, encoding='bgr8'))

        # ── Occupancy grid ────────────────────────────────────────────────
        self.pub_occ.publish(
            self._to_ros_occ_grid(occ_grid, stamp))

        # ── Coefficients ──────────────────────────────────────────────────
        m             = self._metrics
        dim           = MultiArrayDimension(
            label='coefficients', size=3, stride=3)
        coeff_msg             = Float32MultiArray()
        coeff_msg.layout.dim  = [dim]
        coeff_msg.data        = [
            float(m['A']), float(m['B']), float(m['C'])
        ]
        self.pub_coeffs.publish(coeff_msg)

        # ── Status JSON ───────────────────────────────────────────────────
        s_msg      = String()
        s_msg.data = (
            f'{{"status":"{m["status"]}",'
            f'"pts":{m["pts"]},'
            f'"center_x":{m["center_x"]},'
            f'"A":{m["A"]:.6f},'
            f'"B":{m["B"]:.4f},'
            f'"C":{m["C"]:.2f},'
            f'"lateral_jump":{m["lateral_jump"]:.3f},'
            f'"curv_jump":{m["curv_jump"]:.5f},'
            f'"recovery":{str(m["recovery"]).lower()}}}'
        )
        self.pub_status.publish(s_msg)

        # ── Optional local windows ────────────────────────────────────────
        if self.show_win:
            cv2.imshow('sumit_laneCV | Final',
                       cv2.resize(result, (1280, 720)))
            cv2.imshow('sumit_laneCV | Debug',     dbg)
            cv2.imshow('sumit_laneCV | Occupancy', occ_display)
            cv2.waitKey(1)

    # ──────────────────────────────────────────────────────────────────────
    #  ROS OccupancyGrid builder
    # ──────────────────────────────────────────────────────────────────────

    def _to_ros_occ_grid(self,
                         occ_grid: np.ndarray,
                         stamp) -> OccupancyGrid:
        msg                           = OccupancyGrid()
        msg.header.stamp              = stamp
        msg.header.frame_id           = 'base_footprint_imu'
        msg.info.resolution           = (
            float(UNWARPED_SIZE[0]) / OCC_COLS / 1000.0
        )
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
    #  Histogram centre finder
    # ──────────────────────────────────────────────────────────────────────

    def _find_center_x(self, filtered: np.ndarray):
        hist   = np.sum(filtered, axis=0).astype(np.float32)
        window = (SEARCH_WINDOW_RECOVERY
                  if self._recovery_mode else SEARCH_WINDOW)

        if self._last_center_x is not None:
            lo       = max(0, self._last_center_x - window)
            hi       = min(UNWARPED_SIZE[0], self._last_center_x + window)
            windowed = np.zeros_like(hist)
            windowed[lo:hi] = hist[lo:hi]
            if windowed.max() >= 5:
                hist = windowed

        if hist.max() < 5:
            return self._last_center_x

        cx                  = int(np.argmax(hist))
        self._last_center_x = cx
        return cx

    # ──────────────────────────────────────────────────────────────────────
    #  Core detection pipeline
    # ──────────────────────────────────────────────────────────────────────

    def _detect_lane(self, frame: np.ndarray):

        if (frame.shape[1], frame.shape[0]) != ORIGINAL_SIZE:
            frame = cv2.resize(frame, ORIGINAL_SIZE)

        undist = cv2.undistort(frame, K, D)
        warped = cv2.warpPerspective(
            undist, M, UNWARPED_SIZE, flags=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        enhanced, edges_gated, tape_mask = preprocess_lane(gray)

        roi = edges_gated.copy()
        roi[:ROI_TOP,  :]  = 0
        roi[:,  :ROI_LEFT] = 0
        roi[:, ROI_RIGHT:] = 0

        filtered = filter_dashes(roi)
        center_x = self._find_center_x(filtered)

        if center_x is not None:
            win    = (SEARCH_WINDOW_RECOVERY
                      if self._recovery_mode else SEARCH_WINDOW)
            locked = np.zeros_like(filtered)
            lo     = max(0, center_x - win)
            hi     = min(UNWARPED_SIZE[0], center_x + win)
            locked[:, lo:hi] = filtered[:, lo:hi]
        else:
            locked = filtered

        ys, xs   = np.where(locked > 0)
        lat_jump = curv_jump = 0.0
        status   = 'INIT'
        coeffs   = None

        if len(ys) < 20:
            self._consecutive_rejects += 1
            status = f'LOW_PTS({len(ys)})'
            if self._consecutive_rejects >= RECOVERY_THRESHOLD:
                self._recovery_mode  = True
                self._last_center_x  = None
            if self._coeff_history:
                coeffs = np.mean(self._coeff_history, axis=0)

        else:
            new_c = np.polyfit(ys, xs, 2)
            valid, lat_jump, curv_jump = is_valid_fit(
                new_c, self._coeff_history)

            if valid or self._recovery_mode:
                self._coeff_history.append(new_c)
                self._consecutive_rejects = 0
                self._recovery_mode       = False
                status = f'OK pts={len(ys)}'
            else:
                self._consecutive_rejects += 1
                status = f'REJ pts={len(ys)}'
                if self._consecutive_rejects >= RECOVERY_THRESHOLD:
                    self._recovery_mode  = True
                    self._last_center_x  = None

            if self._coeff_history:
                coeffs = np.mean(self._coeff_history, axis=0)

        self._metrics.update({
            'status':       status,
            'pts':          len(ys),
            'center_x':     center_x if center_x is not None else -1,
            'lateral_jump': lat_jump,
            'curv_jump':    curv_jump,
            'A':  float(coeffs[0]) if coeffs is not None else 0.0,
            'B':  float(coeffs[1]) if coeffs is not None else 0.0,
            'C':  float(coeffs[2]) if coeffs is not None else 0.0,
            'smooth_n':  len(self._coeff_history),
            'recovery':  self._recovery_mode,
        })

        ploty = np.linspace(ROI_TOP, UNWARPED_SIZE[1] - 1, 200)
        center_fit = (np.polyval(coeffs, ploty)
                      if coeffs is not None
                      else np.full_like(ploty, UNWARPED_SIZE[0] / 2.0))

        left_x  = center_fit - LANE_OFFSET
        right_x = center_fit + LANE_OFFSET
        l_valid = (left_x  >= 0) & (left_x  < UNWARPED_SIZE[0])
        r_valid = (right_x >= 0) & (right_x < UNWARPED_SIZE[0])

        occ_grid    = build_occupancy_grid(coeffs, left_x, right_x)
        occ_display = render_occupancy_grid(occ_grid, self._metrics)

        fit_vis = np.zeros(
            (UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)
        fit_vis[ys, xs] = (0, 0, 255)
        for i in range(len(ploty)):
            cy = int(ploty[i])
            cx = int(center_fit[i])
            if 0 <= cx < UNWARPED_SIZE[0]:
                cv2.circle(fit_vis, (cx, cy), 2, (0, 255, 255), -1)
            if l_valid[i]:
                cv2.circle(fit_vis,
                           (int(left_x[i]),  cy), 2, (255, 0,   0), -1)
            if r_valid[i]:
                cv2.circle(fit_vis,
                           (int(right_x[i]), cy), 2, (0,   0, 255), -1)

        overlay = np.zeros(
            (UNWARPED_SIZE[1], UNWARPED_SIZE[0], 3), dtype=np.uint8)

        pts_l = np.array(
            [[int(left_x[i]),  int(ploty[i])]
             for i in range(len(ploty)) if l_valid[i]],
            dtype=np.int32)
        pts_r = np.array(
            [[int(right_x[i]), int(ploty[i])]
             for i in range(len(ploty)) if r_valid[i]],
            dtype=np.int32)

        if len(pts_l) > 2 and len(pts_r) > 2:
            fill = np.concatenate([pts_l, np.flipud(pts_r)], axis=0)
            cv2.fillPoly(overlay, [fill], (0, 180, 0))
        if len(pts_l) > 1:
            cv2.polylines(overlay, [pts_l], False, (255, 0,   0), 4)
        if len(pts_r) > 1:
            cv2.polylines(overlay, [pts_r], False, (0,   0, 255), 4)

        pts_c = np.array(
            [[int(center_fit[i]), int(ploty[i])]
             for i in range(len(ploty))
             if 0 <= int(center_fit[i]) < UNWARPED_SIZE[0]],
            dtype=np.int32)
        if len(pts_c) > 1:
            cv2.polylines(overlay, [pts_c], False, (0, 255, 255), 3)

        overlay_inv = cv2.warpPerspective(
            overlay, M_inv, ORIGINAL_SIZE, flags=cv2.INTER_LINEAR)
        result = cv2.addWeighted(undist, 1.0, overlay_inv, 0.5, 0)
        result = draw_metrics(result, self._metrics)

        filtered_vis = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(filtered_vis,
                      (ROI_LEFT, ROI_TOP),
                      (ROI_RIGHT, UNWARPED_SIZE[1] - 1),
                      (0, 255, 255), 2)

        dbg = build_debug_grid(
            undist, warped, enhanced, tape_mask,
            edges_gated, filtered_vis, fit_vis, occ_display)

        return result, dbg, occ_display, occ_grid

    # ──────────────────────────────────────────────────────────────────────
    #  Cleanup
    # ──────────────────────────────────────────────────────────────────────

    def destroy_node(self):
        if self._pipeline is not None and self._pipeline.isRunning():
            # v3 API cleanup — context manager handles device close
            pass
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