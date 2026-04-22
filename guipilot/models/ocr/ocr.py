import base64
import os

import cv2
import requests
import numpy as np
from paddleocr import PaddleOCR 


class OCR():
    def __init__(self, service_url: str = None) -> None:
        self.service_url = service_url

        if service_url is None:
            import paddle
            device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
            lang = os.environ.get("OCR_LANG", "ch")
            self.ocr = PaddleOCR(
                lang=lang,
                device=device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

    def _local(self, image: np.ndarray) -> tuple[list, list]:
        texts, text_bboxes = [], []
        for page in self.ocr.predict(image):
            for text, box in zip(page.get("rec_texts", []), page.get("rec_boxes", [])):
                xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                texts.append(text)
                text_bboxes.append([xmin, ymin, xmax, ymax])
        return texts, text_bboxes

    def __call__(self, image: np.ndarray) -> tuple[list, list]:
        if self.service_url is None: return self._local(image)

        _, buffer = cv2.imencode(".jpg", image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        response = requests.post(self.service_url, data={"image_array": img_base64})
        data: dict = response.json()
        texts = data.get("text")
        text_bboxes = data.get("box")

        return texts, text_bboxes