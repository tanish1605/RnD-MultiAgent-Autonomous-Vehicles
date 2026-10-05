#!/usr/bin/env python3

"""
detector.py

YOLOv8 detector wrapper.

Responsibilities
----------------
1. Load YOLO model
2. Run inference
3. Return clean detection dictionaries

No ROS code should exist here.
"""

from ultralytics import YOLO
import torch


class Detector:

    def __init__(
        self,
        model_path,
        confidence=0.50,
        iou=0.45,
        image_size=640,
        device=None
    ):

        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size

        # Automatically choose GPU if available
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print("=" * 60)
        print("Loading YOLO Model")
        print("Model :", model_path)
        print("Device:", self.device)
        print("=" * 60)

        self.model = YOLO(model_path)

        # Move model to selected device
        self.model.to(self.device)

        self.class_names = self.model.names

        print("Model Loaded Successfully.")
        print("Classes:", len(self.class_names))
        print("=" * 60)

    def detect(self, frame):
        """
        Run inference on one BGR image.

        Returns
        -------
        List[dict]
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            verbose=False,
            device=self.device
        )

        detections = []

        if len(results) == 0:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:

            cls_id = int(box.cls.item())

            conf = float(box.conf.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({

                "class_id": cls_id,

                "class_name": self.class_names[cls_id],

                "confidence": conf,

                "bbox": [

                    int(x1),

                    int(y1),

                    int(x2),

                    int(y2)

                ]
            })

        return detections
