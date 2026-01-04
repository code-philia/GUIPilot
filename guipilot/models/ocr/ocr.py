import base64
import os

import cv2
import requests
import numpy as np


class OCR():
    def __init__(
        self,
        service_url: str | None = None,
        use_gpu: bool | None = None,
        language: str | None = None,
        enable_local: bool | None = None,
    ) -> None:
        self.service_url = service_url
        self.language = language or os.getenv("PADDLEOCR_LANG", "ch")
        self._local_ocr = None

        if enable_local is None:
            enable_local = os.getenv("ENABLE_PADDLEOCR", "0") == "1"

        if service_url is None and enable_local:
            if use_gpu is None:
                use_gpu = os.getenv("PADDLEOCR_USE_GPU", "0") == "1"

            try:
                from paddleocr import PaddleOCR  # pylint: disable=import-error
            except Exception as exc:
                print(f"[warning] PaddleOCR 未安装或导入失败：{exc}")
                return

            try:
                self._local_ocr = PaddleOCR(lang=self.language, show_log=False, use_gpu=use_gpu)
            except Exception as exc:
                if use_gpu:
                    try:
                        self._local_ocr = PaddleOCR(lang=self.language, show_log=False, use_gpu=False)
                    except Exception as cpu_exc:
                        print(f"[warning] PaddleOCR 初始化失败（CPU 回退亦失败）：{cpu_exc}")
                        self._local_ocr = None
                else:
                    print(f"[warning] PaddleOCR 初始化失败：{exc}")
                    self._local_ocr = None

    def _local(self, image: np.ndarray) -> tuple[list, list]:
        texts, text_bboxes = [], []
        if self._local_ocr is None:
            return texts, text_bboxes

        result = self._local_ocr.ocr(image, cls=False)
        for line in result:
            if not line: continue
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
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        response = requests.post(self.service_url, data={"image_array": img_base64})
        data: dict = response.json()
        texts = data.get("text")
        text_bboxes = data.get("box")

        return texts, text_bboxes