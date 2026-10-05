#!/usr/bin/env python3
import cv2
import time
import os
import queue
import threading
import subprocess
import sys
import depthai as dai

# --- Global Configuration ---
ORIGINAL_SIZE = (1920, 1080)
FPS = 30

# --- Global Control Flags ---
is_recording = False
filename = ""
rec_start_time = 0
frames_recorded = 0

# --- Writer thread globals ---
write_queue    = None
writer_thread  = None
stop_event_ref = None
frames_written = 0


# ---------------------------------------------------------------------------
# FFmpeg writer thread - replaces cv2.VideoWriter
# ---------------------------------------------------------------------------
def ffmpeg_writer_worker(filepath, width, height, fps, q, stop_evt):
    global frames_written
    frames_written = 0

    cmd = [
        "ffmpeg", "-y",
        "-f",            "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size",   f"{width}x{height}",
        "-framerate",    str(fps),
        "-i",            "pipe:0",
        "-vcodec",       "libx264",
        "-preset",       "ultrafast",
        "-crf",          "18",
        "-pix_fmt",      "yuv420p",
        "-movflags",     "+faststart",
        filepath
    ]

    print(f"[FFMPEG] {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        print("[FFMPEG] ERROR: ffmpeg not found. Run: sudo apt install ffmpeg")
        return

    expected_bytes = width * height * 3

    while True:
        if stop_evt.is_set() and q.empty():
            break
        try:
            frame_bgr, _ = q.get(timeout=0.1)
        except queue.Empty:
            continue

        # Verify frame shape before writing - mismatch = corrupted file
        if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
            print(f"[FFMPEG] Shape mismatch: {frame_bgr.shape}, skipping")
            continue

        raw = frame_bgr.tobytes()
        if len(raw) != expected_bytes:
            print(f"[FFMPEG] Byte count wrong: {len(raw)}, skipping")
            continue

        try:
            proc.stdin.write(raw)
            frames_written += 1
        except (BrokenPipeError, OSError) as e:
            print(f"[FFMPEG] Pipe error: {e}")
            break

    # Flush encoder - communicate() closes stdin internally, don't close manually
    print(f"[FFMPEG] Flushing encoder ({frames_written} frames)...")
    try:
        _, stderr_bytes = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        _, stderr_bytes = proc.communicate()

    if proc.returncode != 0:
        lines = stderr_bytes.decode(errors="ignore").strip().split("\n")
        print("[FFMPEG] Error output:")
        for line in lines[-15:]:
            print(f"    {line}")
    else:
        if os.path.exists(filepath):
            mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"[FFMPEG] Saved: {filepath} ({mb:.1f} MB, {frames_written} frames)")


# ---------------------------------------------------------------------------
# Start / Stop helpers
# ---------------------------------------------------------------------------
def start_recording():
    global is_recording, write_queue, writer_thread, stop_event_ref
    global rec_start_time, frames_recorded, frames_written, filename

    if is_recording:
        return

    filename = f"lane_test_{int(time.time())}.mp4"
    frames_recorded = 0
    frames_written  = 0
    rec_start_time  = time.time()

    write_queue    = queue.Queue(maxsize=90)   # 3s buffer at 30fps
    stop_event_ref = threading.Event()

    writer_thread = threading.Thread(
        target=ffmpeg_writer_worker,
        args=(filename,
              ORIGINAL_SIZE[0], ORIGINAL_SIZE[1],
              FPS, write_queue, stop_event_ref),
        daemon=True,
        name="FFmpegWriter"
    )
    writer_thread.start()
    is_recording = True
    print(f"[INFO] Recording started: {filename}")


def stop_recording():
    global is_recording

    if not is_recording:
        return

    is_recording = False

    if stop_event_ref:
        stop_event_ref.set()

    if writer_thread and writer_thread.is_alive():
        print("[INFO] Waiting for encoder to finish...")
        writer_thread.join(timeout=60)

    dur  = time.time() - rec_start_time
    size = os.path.getsize(filename) / (1024*1024) if os.path.exists(filename) else 0
    print(f"[INFO] Saved {filename} "
          f"({dur:.1f}s | {frames_recorded} captured "
          f"| {frames_written} encoded | {size:.1f} MB)")


# ---------------------------------------------------------------------------
# Mouse callback - YOUR original coordinates kept exactly
# ---------------------------------------------------------------------------
def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # START button (X: 50-200, Y: 735-785)
        if 50 <= x <= 200 and 735 <= y <= 785:
            start_recording()

        # STOP button (X: 230-380, Y: 735-785)
        elif 230 <= x <= 380 and 735 <= y <= 785:
            stop_recording()


# ---------------------------------------------------------------------------
# DepthAI pipeline - YOUR original code, untouched
# ---------------------------------------------------------------------------
def build_depthai_pipeline(fps: int):
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.Camera).build(
        boardSocket=dai.CameraBoardSocket.CAM_A,
        sensorResolution=(ORIGINAL_SIZE[0], ORIGINAL_SIZE[1]),
        sensorFps=float(fps)
    )

    video_out = cam.requestOutput(
        size=(ORIGINAL_SIZE[0], ORIGINAL_SIZE[1]),
        type=dai.ImgFrame.Type.BGR888p
    )

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.addRotateDeg(180)
    manip.setMaxOutputFrameSize(ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1] * 3)
    video_out.link(manip.inputImage)

    video_queue = manip.out.createOutputQueue(maxSize=8, blocking=False)

    pipeline.start()

    print(
        f"[DepthAI] OAK camera started "
        f"({ORIGINAL_SIZE[0]}x{ORIGINAL_SIZE[1]} @ {fps} fps) "
        f"API {dai.__version__}"
    )

    return pipeline, video_queue


# ---------------------------------------------------------------------------
# Main - YOUR original structure, only writer.write() swapped out
# ---------------------------------------------------------------------------
def main():
    global frames_recorded

    pipeline, video_queue = build_depthai_pipeline(FPS)

    window_name = "OAK-D Jetson Recorder (FFmpeg)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_handler)

    print("UI Ready. Click START/STOP or press R/S. Press Q to quit.")

    try:
        while pipeline.isRunning():
            if video_queue.has():
                pkt   = video_queue.get()
                frame = pkt.getCvFrame()

                # --- THE ONLY CHANGE FROM YOUR LAPTOP SCRIPT ---
                # Before: writer.write(frame)  ← CPU bottleneck on Jetson
                # After:  non-blocking queue push to FFmpeg thread
                if is_recording and write_queue is not None:
                    try:
                        write_queue.put_nowait((frame.copy(), time.monotonic()))
                        frames_recorded += 1
                    except queue.Full:
                        # Writer falling behind - drop frame rather than stall camera
                        print(f"[WARN] Queue full, dropped frame {frames_recorded}")

                # --- YOUR original HUD code, untouched ---
                canvas = cv2.copyMakeBorder(
                    frame, 0, 80, 0, 0,
                    cv2.BORDER_CONSTANT, value=[30, 30, 30]
                )

                start_color = (0, 180, 0) if not is_recording else (50, 80, 50)
                stop_color  = (0, 0, 180) if is_recording     else (50, 50, 80)

                cv2.rectangle(canvas, (50, 735),  (200, 785), start_color, -1)
                cv2.putText(canvas, "START", (88, 767),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                cv2.rectangle(canvas, (230, 785), (380, 785), stop_color, -1)
                cv2.putText(canvas, "STOP", (275, 767),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                status_text  = "RECORDING" if is_recording else "IDLE"
                status_color = (0, 0, 255)  if is_recording else (0, 255, 0)
                cv2.circle(canvas, (430, 760), 8, status_color, -1)
                cv2.putText(canvas, f"STATUS: {status_text}", (455, 767),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                if is_recording:
                    elapsed = time.time() - rec_start_time
                    cv2.putText(
                        canvas,
                        f"{frames_recorded} frames | {frames_written} encoded | {elapsed:5.1f}s",
                        (650, 767),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1
                    )

                cv2.putText(canvas, "Press 'q' to Quit",
                            (1060, 767), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (170,170,170), 1)

                cv2.imshow(window_name, canvas)

            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'):
                break
            elif key == ord('r'):
                start_recording()
            elif key == ord('s'):
                stop_recording()

    finally:
        stop_recording()
        cv2.destroyAllWindows()
        print("Script terminated clean.")


if __name__ == "__main__":
    main()