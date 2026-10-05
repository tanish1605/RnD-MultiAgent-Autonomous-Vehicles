import cv2
import numpy as np

# ==========================
# INPUT / OUTPUT
# ==========================
input_video = "run3.mp4"
output_video = "run3_output.mp4"

# ==========================
# BEV PARAMETERS
# ==========================
bev_width = 600
bev_height = 800

# Source points in original image
src_points = np.array([
    (675, 543),
    (1247, 517),
    (1883, 851),
    (43,  882)
], dtype=np.float32)

# Destination points in BEV image
dst_points = np.float32([
    [0, 0],
    [bev_width, 0],
    [bev_width, bev_height],
    [0, bev_height]
])

# ==========================
# OPEN VIDEO
# ==========================
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {input_video}")

fps = cap.get(cv2.CAP_PROP_FPS)

# Perspective transform matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (bev_width, bev_height)
)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Generate BEV
    bev = cv2.warpPerspective(
        frame,
        M,
        (bev_width, bev_height)
    )

    # Save frame
    writer.write(bev)

    # Optional display
    cv2.imshow("BEV", bev)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Saved {frame_count} frames")
print(f"Output video: {output_video}")