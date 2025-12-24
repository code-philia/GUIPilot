import os
import base64

import cv2
import torch
import requests
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .. import model_manager


class Detector():
    def __init__(self, service_url: str = None) -> None:
        self.service_url = service_url

        if service_url is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            base_path = os.path.dirname(os.path.abspath(__file__))
            # default weights filename next to this file
            weights_path = os.path.join(base_path, "best.pt")
            # allow override via env vars: GUIPILOT_DETECTOR_HF_REPO / GUIPILOT_DETECTOR_HF_FILE
            repo = os.environ.get("GUIPILOT_DETECTOR_HF_REPO")
            hf_file = os.environ.get("GUIPILOT_DETECTOR_HF_FILE")

            try:
                # ensure_detector_weights returns the actual path where the model was saved
                downloaded = model_manager.ensure_detector_weights(target_path=weights_path, hf_repo=repo, hf_filename=hf_file)
                weights_path = downloaded or weights_path
            except Exception as e:
                # If download fails, fall back to package-local best.pt (may raise later when loading)
                print("Model download/ensure failed:", e)

            print(f"Loading detector weights from: {weights_path}")
            self.detector = YOLO(weights_path).to(device)

    def _local(self, image: np.ndarray) -> tuple[list, list]:
        results: list[Results] = self.detector(image, verbose=False)
        bboxes = results[0].boxes.xyxy.cpu().numpy() 
        class_ids = results[0].boxes.cls.cpu().numpy()
        sorted_indices = np.lexsort((bboxes[:, 0], bboxes[:, 1])) 
        sorted_bboxes = bboxes[sorted_indices]
        sorted_class_ids = class_ids[sorted_indices]
        sorted_widget_types = [self.detector.names[int(class_id)] for class_id in sorted_class_ids]
        return sorted_bboxes, sorted_widget_types

    def __call__(self, image: np.ndarray):
        if self.service_url is None: return self._local(image)

        _, buffer = cv2.imencode(".jpg", image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        response = requests.post(self.service_url, data={"image_array": img_base64})
        data: dict = response.json()
        widget_types = np.array(data.get("class"))
        bboxes = np.array(data.get("box"))
        sorted_indices = np.lexsort((bboxes[:, 0], bboxes[:, 1])) 
        sorted_bboxes = bboxes[sorted_indices]
        sorted_widget_types = widget_types[sorted_indices]
        return sorted_bboxes, sorted_widget_types