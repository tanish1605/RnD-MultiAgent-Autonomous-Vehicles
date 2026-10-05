#!/usr/bin/env python3

"""
Visualization utilities.

Draws detections on an image.

No ROS code.
"""

import cv2
from .utils import format_confidence, format_distance

class Visualizer:

    def __init__(self):

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.font_scale = 0.6

        self.thickness = 2

    def _get_color(self, confidence):

        if confidence >= 0.80:
            return (0, 255, 0)

        elif confidence >= 0.60:
            return (0, 255, 255)

        else:
            return (0, 0, 255)

    def draw(self, frame, detections):

        image = frame.copy()

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            conf = det["confidence"]

            cls = det["class_name"]

            color = self._get_color(conf)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            # label = f"{cls} {conf:.2f}"
            label_lines = [
                det["class_name"],
                f"Conf : {format_confidence(det['confidence'])}"
            ]

            distance = det.get("distance")
            if distance is not None:
                label_lines.append(f"Dist : {format_distance(distance)}")

            max_width = 0
            line_height = 20

            for text in label_lines:
                (w, h), _ = cv2.getTextSize(
                    text,
                    self.font,
                    0.5,
                    1,
                )
                max_width = max(max_width, w)

            text_height = len(label_lines) * line_height + 8
            top = max(0, y1 - text_height)

            # Draw background rectangle
            cv2.rectangle(
                image,
                (x1, top),
                (x1 + max_width + 10, y1),
                color,
                -1,
            )

            for i, text in enumerate(label_lines):
                cv2.putText(
                    image,
                    text,
                    (x1 + 5, top + 16 + i * line_height),
                    self.font,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return image
