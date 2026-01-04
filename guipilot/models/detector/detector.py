import base64
import os

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


class Detector:
    def __init__(self, service_url: str = None) -> None:
        self.service_url = service_url

        if service_url is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            base_path = os.path.dirname(os.path.abspath(__file__))
            self.detector = YOLO(f"{base_path}/best.pt").to(device)

    def _local(self, image: np.ndarray) -> tuple[list, list]:
        results: list[Results] = self.detector(image, verbose=False)
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        sorted_indices = np.lexsort((bboxes[:, 0], bboxes[:, 1]))
        sorted_bboxes = bboxes[sorted_indices]
        sorted_class_ids = class_ids[sorted_indices]
        sorted_widget_types = [
            self.detector.names[int(class_id)] for class_id in sorted_class_ids
        ]
        return sorted_bboxes, sorted_widget_types

    def __call__(self, image: np.ndarray):
        if self.service_url is None:
            return self._local(image)

        _, buffer = cv2.imencode(".jpg", image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        response = requests.post(self.service_url, data={"image_array": img_base64})
        data: dict = response.json()
        widget_types = np.array(data.get("class"))
        bboxes = np.array(data.get("box"))
        sorted_indices = np.lexsort((bboxes[:, 0], bboxes[:, 1]))
        sorted_bboxes = bboxes[sorted_indices]
        sorted_widget_types = widget_types[sorted_indices]
        return sorted_bboxes, sorted_widget_types
