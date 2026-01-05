import base64

import cv2
import numpy as np
import requests
from paddleocr import PaddleOCR


class OCR:
    def __init__(self, service_url: str = None) -> None:
        self.service_url = service_url

        if service_url is None:
            self.ocr = PaddleOCR(lang="ch", show_log=False, use_gpu=True)

    def _local(self, image: np.ndarray) -> tuple[list, list]:
        texts, text_bboxes = [], []
        result = self.ocr.ocr(image, cls=False)
        for line in result:
            if not line:
                continue
            for word_info in line:
                bbox = word_info[0]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                bbox = [xmin, ymin, xmax, ymax]
                text = word_info[1][0]
                texts.append(text)
                text_bboxes.append(bbox)

        return texts, text_bboxes

    def __call__(self, image: np.ndarray) -> tuple[list, list]:
        if self.service_url is None:
            return self._local(image)

        _, buffer = cv2.imencode(".jpg", image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        response = requests.post(self.service_url, data={"image_array": img_base64})
        data: dict = response.json()
        texts = data.get("text")
        text_bboxes = data.get("box")

        return texts, text_bboxes
