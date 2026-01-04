from __future__ import annotations
import typing
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .constants import Bbox
from .widget import Widget, WidgetType
from guipilot.models import OCR, Detector

if typing.TYPE_CHECKING:
    from .screen import Screen
    from guipilot.checker import ScreenChecker
    from guipilot.matcher import WidgetMatcher


OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL")
DETECTOR_SERVICE_URL = os.getenv("DETECTOR_SERVICE_URL")

ocr = OCR(service_url=OCR_SERVICE_URL)
detector = Detector(service_url=DETECTOR_SERVICE_URL)


@dataclass
class Screen:
    image: np.ndarray
    widgets: dict[int, Widget] = field(default_factory=dict)
    
    def detect(self) -> None:
        """
        Use object detector to extract widgets from screen
        Updates widgets list to the detection results
        """
        def _to_bbox(points: np.ndarray) -> Bbox:
            xmin, ymin, xmax, ymax = points
            return Bbox(int(xmin), int(ymin), int(xmax), int(ymax))
        
        bboxes, widget_types = detector(self.image)
        self.widgets = {
            i: Widget(
                type=WidgetType(widget_type),
                bbox=_to_bbox(bbox)
            )
            for i, (bbox, widget_type) in enumerate(zip(bboxes, widget_types))
        }

        assert len(self.widgets) == len(bboxes) == len(widget_types)

    def ocr(self) -> None:
        """
        Use OCR to extract text from screen and assign to widgets
        """
        image = np.array(self.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        widgets = self.widgets.values()
        for widget in widgets:
            xmin, ymin, xmax, ymax = widget.bbox
            widget_image = image[ymin:ymax, xmin:xmax].copy()
            try:
                widget.texts, widget.text_bboxes = ocr(widget_image)
            except Exception as e:
                print(e)
                print(self.image.shape)
                print(widget.bbox)

    def check(self, target: Screen, matcher: WidgetMatcher, checker: ScreenChecker) -> tuple[set, float]:
        """Check for screen inconsistency

        Args:
            target: the screen to check against
            matcher: algorithm to match same widgets as pairs on both screens
            checker: algorithm to check consistency of widget pairs

        Returns:
            see ScreenChecker
        """
        pairs = matcher.match(self, target)
        inconsistencies, time_taken = checker.check(self, target, pairs)
        return inconsistencies, time_taken