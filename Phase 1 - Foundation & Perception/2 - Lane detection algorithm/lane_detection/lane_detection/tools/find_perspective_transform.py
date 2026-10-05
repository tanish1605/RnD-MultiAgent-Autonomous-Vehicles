import cv2
import numpy as np


# ── OAK-D Intrinsics (CAM_B / Left Cam) ──────────────────────────────────
K = np.array([
    [796.8074340820312, 0.0,               632.9873046875],
    [0.0,               796.5176391601562, 431.61431884765625],
    [0.0,               0.0,               1.0]
], dtype=np.float64)

D = np.array([
    24.26119041442871, -44.0109977722168, -0.001162048545666039,
    -4.417484433361096e-06, 26.147001266479492, 23.662704467773438,
    -42.24923324584961, 24.82693099975586,
    0.0, 0.0, 0.0, 0.0,
    -0.010093020275235176, -0.008346530608832836
], dtype=np.float64)


camera_matrix = K

distortion_coefficients = D

image = cv2.imread(
    "/home/sumit/all_lane_detection/data/test_images/test8.png"
)

if image is None:
    print("Image not found!")

    exit()

undistorted = cv2.undistort(
    image,
    camera_matrix,
    distortion_coefficients
)
height, width = undistorted.shape[:2]

src = np.float32([

    [410, 448],   # top-left

    [1474, 446],   # top-right

    [1546, 875],  # bottom-right

    [352, 880]    # bottom-left

])

dst = np.array(
    [(0, 0), (500, 0), (500, 600), (0, 600)],
    dtype=np.float32
)

M     = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

M = cv2.getPerspectiveTransform(
    src,
    dst
)

warped = cv2.warpPerspective(

    undistorted,
    M,
    (width, height)
)

debug_image = undistorted.copy()

cv2.polylines(

    debug_image,
    [src.astype(np.int32)],
    True,
    (0, 255, 0),
    3
)

cv2.imshow(
    "Source Region",
    debug_image
)

cv2.imshow(
    "Warped Bird Eye View",
    warped
)

cv2.waitKey(0)

cv2.destroyAllWindows()

perspective_data = {

    "perspective_matrix": M,

    "src_points": src,

    "dst_points": dst

}