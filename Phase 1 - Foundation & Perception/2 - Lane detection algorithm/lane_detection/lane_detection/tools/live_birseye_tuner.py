#!/usr/bin/env python3
import cv2
import numpy as np
import depthai as dai

# ── 1. Camera Intrinsics (CAM_A) ─────────────────────────────────────────
WIDTH, HEIGHT = 1920, 1080

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

# ── 2. Global State for Interactive Tuning ───────────────────────────────
# Initial guess for [TL, TR, BR, BL]
src_points = np.float32([
    [538, 641],   # Top-Left
    [1382, 610],  # Top-Right
    [1892, 784],  # Bottom-Right
    [26, 805]    # Bottom-Left
])

# The output canvas size you requested earlier
DST_WIDTH, DST_HEIGHT = 500, 600
dst_points = np.float32([
    [0, 0], [DST_WIDTH, 0], [DST_WIDTH, DST_HEIGHT], [0, DST_HEIGHT]
])

# Interaction variables
dragging_point_idx = None
point_labels = ["TL", "TR", "BR", "BL"]

# ── 3. Mouse Interaction Logic ───────────────────────────────────────────
def mouse_handler(event, x, y, flags, param):
    global src_points, dragging_point_idx
    
    # Left click down: check if we clicked near any of the 4 points
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(src_points):
            # 30 pixel grab radius
            if np.linalg.norm(pt - np.array([x, y])) < 30:
                dragging_point_idx = i
                break
                
    # Mouse move: update the coordinates of the point we are dragging
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point_idx is not None:
            # Bound the coordinates to the screen size
            new_x = max(0, min(WIDTH, x))
            new_y = max(0, min(HEIGHT, y))
            src_points[dragging_point_idx] = [new_x, new_y]
            
    # Left click release: stop dragging
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point_idx = None

# ── 4. DepthAI Pipeline ──────────────────────────────────────────────────
def build_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A,
        sensorResolution=(WIDTH, HEIGHT),
        sensorFps=30
    )
    video_out = cam.requestOutput((WIDTH, HEIGHT), type=dai.ImgFrame.Type.BGR888p)
    
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.addRotateDeg(180)
    manip.setMaxOutputFrameSize(WIDTH * HEIGHT * 3)
    video_out.link(manip.inputImage)
    
    queue = manip.out.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()
    return pipeline, queue

# ── 5. Main Loop ─────────────────────────────────────────────────────────
def main():
    pipeline, queue = build_pipeline()
    
    # Setup OpenCV windows
    window_main = "Interactive Tuner - Drag the Red Dots!"
    window_warp = "Real-Time Bird Eye View"
    
    # Using WINDOW_NORMAL allows us to resize the window visually on your monitor 
    # WITHOUT messing up the mouse coordinates, which stay mapped to the 1080p frame!
    cv2.namedWindow(window_main, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_main, 1280, 720) 
    cv2.setMouseCallback(window_main, mouse_handler)

    print("\n" + "="*50)
    print(" LIVE TUNER STARTED")
    print(" 1. Click and drag the red dots in the main window.")
    print(" 2. Watch the Bird's Eye View update instantly.")
    print(" 3. Press 'q' to quit and print the final coordinates.")
    print("="*50 + "\n")

    while True:
        if queue.has():
            pkt = queue.get()
            frame = pkt.getCvFrame()

            # 1. Undistort the raw frame
            undistorted = cv2.undistort(frame, K, D)
            
            # 2. Calculate perspective matrix using CURRENT live points
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # 3. Warp the image
            warped = cv2.warpPerspective(undistorted, M, (DST_WIDTH, DST_HEIGHT))

            # 4. Draw UI on the main frame
            debug_frame = undistorted.copy()
            
            # Draw the green polygon connecting the points
            cv2.polylines(debug_frame, [src_points.astype(np.int32)], True, (0, 255, 0), 3)
            
            # Draw the grab handles (red circles) and labels
            for i, pt in enumerate(src_points):
                x, y = int(pt[0]), int(pt[1])
                # Highlight the circle if it's currently being dragged
                color = (0, 255, 255) if i == dragging_point_idx else (0, 0, 255)
                
                cv2.circle(debug_frame, (x, y), 15, color, -1)
                cv2.circle(debug_frame, (x, y), 15, (255, 255, 255), 2) # white border
                
                cv2.putText(debug_frame, point_labels[i], (x + 20, y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # 5. Display windows
            cv2.imshow(window_main, debug_frame)
            cv2.imshow(window_warp, warped)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    
    # Print the final result formatted perfectly for copy-pasting
    print("\n" + "="*50)
    print("🎯 PERFECT TUNE FOUND! Copy this into your main script:")
    print("="*50)
    print("SRC = np.array([")
    print(f"    ({int(src_points[0][0])}, {int(src_points[0][1])}),  # TL")
    print(f"    ({int(src_points[1][0])}, {int(src_points[1][1])}),  # TR")
    print(f"    ({int(src_points[2][0])}, {int(src_points[2][1])}),  # BR")
    print(f"    ({int(src_points[3][0])}, {int(src_points[3][1])})   # BL")
    print("], dtype=np.float32)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
