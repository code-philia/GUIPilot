from __future__ import annotations

import re
import typing
from difflib import SequenceMatcher

import cv2
import numpy as np

from guipilot.entities import Inconsistency, WidgetType

from .checker import ScreenChecker

if typing.TYPE_CHECKING:
    from guipilot.entities import Widget


class GUIPilot(ScreenChecker):
    def check_widget_pair(
        self, w1: Widget, w2: Widget, wi1: np.ndarray, wi2: np.ndarray
    ) -> list[tuple]:
        def check_bbox_consistency(w1: Widget, w2: Widget) -> bool:
            """Check if both widgets have similar position, size, and shape on the screen"""
            xa, ya = max(w1.bbox[0], w2.bbox[0]), max(w1.bbox[1], w2.bbox[1])
            xb, yb = min(w1.bbox[2], w2.bbox[2]), min(w1.bbox[3], w2.bbox[3])
            intersection = abs(max((xb - xa, 0)) * max((yb - ya), 0))
            boxa = abs((w1.bbox[2] - w1.bbox[0]) * (w1.bbox[3] - w1.bbox[1]))
            boxb = abs((w2.bbox[2] - w2.bbox[0]) * (w2.bbox[3] - w2.bbox[1]))
            iou = intersection / (boxa + boxb - intersection)
            return iou > 0.9

        def check_text_consistency(w1: Widget, w2: Widget) -> bool:
            """Check if the text on both widgets are similar"""
            has_text = {
                WidgetType.TEXT_VIEW,
                WidgetType.TEXT_BUTTON,
                WidgetType.COMBINED_BUTTON,
                WidgetType.INPUT_BOX,
            }
            if w1.type not in has_text or w2.type not in has_text:
                return True

            for t1, t2 in zip(w1.texts, w2.texts):
                t1 = re.sub(r"[^a-zA-Z0-9]", "", t1)
                t2 = re.sub(r"[^a-zA-Z0-9]", "", t2)
                if SequenceMatcher(None, t1.lower(), t2.lower()).quick_ratio() < 0.95:
                    return False

            return True

        def check_color_consistency(wi1: np.ndarray, wi2: np.ndarray) -> bool:
            """Check if the color distribution on both widgets are similar"""
            # normalized 3D color histogram, 8 bins per channel
            hist1 = cv2.calcHist(
                [wi1], [0, 1, 2], None, [8, 8, 8], [0, 250, 0, 250, 0, 250]
            )
            hist1 = cv2.normalize(hist1, hist1).flatten()

            hist2 = cv2.calcHist(
                [wi2], [0, 1, 2], None, [8, 8, 8], [0, 250, 0, 250, 0, 250]
            )
            hist2 = cv2.normalize(hist2, hist2).flatten()

            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_KL_DIV)
            return score < 8

        diff = set()
        if not check_bbox_consistency(w1, w2):
            diff.add(Inconsistency.BBOX)
        if not check_text_consistency(w1, w2):
            diff.add(Inconsistency.TEXT)
        if Inconsistency.TEXT not in diff:
            if not check_color_consistency(wi1, wi2):
                diff.add(Inconsistency.COLOR)

        return list(diff)
