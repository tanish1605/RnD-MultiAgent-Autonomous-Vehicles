import cv2
import numpy as np

image = cv2.imread(
    "/home/sumit/all_lane_detection/data/test_images/test10.png"
)

points = []

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x, y))

        print(f"Point selected: {(x, y)}")

        cv2.circle(
            image,
            (x, y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.imshow(
            "Select Points",
            image
        )

cv2.imshow(
    "Select Points",
    image
)

cv2.setMouseCallback(
    "Select Points",
    mouse_callback
)

cv2.waitKey(0)

cv2.destroyAllWindows()

print("\nSelected Points:")
print(points)