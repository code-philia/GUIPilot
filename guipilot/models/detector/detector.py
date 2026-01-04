import base64
import os
from typing import Tuple

import cv2
import numpy as np
import requests


class Detector:
    """Widget detector used by GUIPilot.

    支持三种模式：
    1. 远程服务 (`DETECTOR_SERVICE_URL`)
    2. 本地 YOLO 權重 (`ENABLE_LOCAL_DETECTOR=1` 且提供權重)
    3. 未配置時返回空結果，避免阻塞流程 / CI。
    """

    def __init__(
        self,
        service_url: str | None = None,
        weight_path: str | None = None,
        enable_local: bool | None = None,
    ) -> None:
        self.service_url = service_url or os.getenv("DETECTOR_SERVICE_URL")
        self._local_detector = None
        self._class_names: dict[int, str] | list[str] = []

        if enable_local is None:
            enable_local = os.getenv("ENABLE_LOCAL_DETECTOR", "0") == "1"

        if weight_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(base_path, "best.pt")
            weight_path = os.getenv("DETECTOR_WEIGHT_PATH", default_path)

        self.weight_path = weight_path

        if self.service_url is None and enable_local:
            self._init_local_detector(weight_path)

    def _init_local_detector(self, weight_path: str) -> None:
        if not os.path.exists(weight_path):
            print(f"[warning] Detector 权重文件缺失：{weight_path}，回退为空结果。")
            return

        try:
            import torch  # pylint: disable=import-error
            from ultralytics import YOLO  # pylint: disable=import-error
        except Exception as exc:  # pragma: no cover
            print(f"[warning] 导入 YOLO 依赖失败：{exc}")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._local_detector = YOLO(weight_path).to(device)
            self._class_names = self._local_detector.names
        except Exception as exc:  # pragma: no cover
            print(f"[warning] 加载 YOLO 权重失败：{exc}")
            self._local_detector = None

    def _empty(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.empty((0, 4), dtype=float), np.empty((0,), dtype=int)

    def _local(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._local_detector is None:
            return self._empty()

        results = self._local_detector(image, verbose=False)
        if not results:
            return self._empty()

        # YOLO 只返回一个 batch
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        if boxes.size == 0:
            return self._empty()

        sorted_indices = np.lexsort((boxes[:, 0], boxes[:, 1]))
        sorted_boxes = boxes[sorted_indices]
        sorted_classes = class_ids[sorted_indices]
        return sorted_boxes, sorted_classes

    def __call__(self, image: np.ndarray):
        if self.service_url is None:
            boxes, class_ids = self._local(image)
        else:
            try:
                boxes, class_ids = self._remote(image)
            except Exception as exc:  # pragma: no cover
                print(f"[warning] 远程检测服务调用失败：{exc}")
                boxes, class_ids = self._empty()

        return self._format_output(boxes, class_ids)

    def _remote(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, buffer = cv2.imencode(".jpg", image)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        response = requests.post(self.service_url, data={"image_array": img_base64}, timeout=30)
        response.raise_for_status()
        data: dict = response.json()
        boxes = np.array(data.get("box") or [])
        class_ids = np.array(data.get("class") or [])
        return boxes, class_ids

    def _format_output(self, boxes: np.ndarray, class_ids: np.ndarray):
        if boxes.size == 0:
            return boxes, []

        if class_ids.size != boxes.shape[0]:
            class_ids = np.resize(class_ids, boxes.shape[0])

        sorted_indices = np.lexsort((boxes[:, 0], boxes[:, 1]))
        sorted_boxes = boxes[sorted_indices]
        sorted_classes = class_ids[sorted_indices]

        widget_types = []
        for cls_id in sorted_classes:
            try:
                widget_types.append(self._class_names[int(cls_id)])
            except Exception:
                widget_types.append(str(int(cls_id)))

        return sorted_boxes, widget_types
