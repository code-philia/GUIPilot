from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from timeit import default_timer as timer

import numpy as np

if typing.TYPE_CHECKING:
    from guipilot.entities import Screen, Widget

import math
import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image

from guipilot.entities import Inconsistency, Widget, WidgetType


class ScreenChecker(ABC):
    def check(
        self, screen_i: Screen, screen_j: Screen, pairs: list[tuple[int, int]]
    ) -> tuple[set, float]:
        """Checks for widget inconsistencies on two screens.

        Args:
            screen_i, screen_j: two screens containing a list of widgets to compare
            pairs: a list of tuples, where each tuple `(x, y)` represents a pair of matching
            widget IDs. `x` is from `screen_i` and `y` is from `screen_j`.

        Returns:
            set: A set of tuples containing:
            - Inconsistent (i, j, type): widget ID pairs with bbox, text, or color inconsistencies
            - Missing (i, None): widget IDs in screen_i that are not paired
            - Excess (None, j): widget IDs in screen_j that are not paired

            float: Time taken (seconds) to check all widgets
        """
        unpaired_i = set(screen_i.widgets.keys())
        unpaired_j = set(screen_j.widgets.keys())

        start_time = timer()
        result = set()
        for pair in pairs:
            x, y = pair
            unpaired_i.discard(x)
            unpaired_j.discard(y)

            widget_i = screen_i.widgets[x]
            xmin, ymin, xmax, ymax = widget_i.bbox
            widget_image_i = screen_i.image[ymin:ymax, xmin:xmax]

            widget_j = screen_j.widgets[y]
            xmin, ymin, xmax, ymax = widget_j.bbox
            widget_image_j = screen_j.image[ymin:ymax, xmin:xmax]

            inconsistencies = self.check_widget_pair(
                widget_i, widget_j, widget_image_i, widget_image_j
            )
            result.update([(x, y, k) for k in inconsistencies])

        result.update([(id, None) for id in unpaired_i])
        result.update([(None, id) for id in unpaired_j])
        time = (timer() - start_time) * 1000
        return result, int(time)

    @abstractmethod
    def check_widget_pair(
        self, w1: Widget, w2: Widget, wi1: np.ndarray, wi2: np.ndarray
    ) -> list[tuple]:
        """Check if a pair of widgets are consistent.

        Args:
            w1, w2: Widget pairs to check
            wi1, wi2: Widget images to check

        Returns:
            A list of tuples, see check() for explanation.
        """
        pass

    def check_text_consistency(self, w1: Widget, w2: Widget) -> bool:
        """
        Compares the similarity of text content between two widgets.

        Only widgets with text-related types (e.g., TEXT_VIEW, INPUT_BOX) are
        evaluated. The algorithm normalizes strings by removing non-alphanumeric
        characters and performs a case-insensitive comparison using SequenceMatcher.

        Args:
            w1 (Widget): First widget containing text strings.
            w2 (Widget): Second widget containing text strings.

        Returns:
            bool: True if text similarity meets the 0.95 threshold or if
                widgets are non-textual.
        """
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

    def check_bbox_consistency(self, w1: Widget, w2: Widget) -> bool:
        """
        Validates if two widgets have similar spatial placement and dimensions.

        Uses the Intersection over Union (IoU) metric to evaluate the overlap
        between two bounding boxes. A high IoU indicates consistent positioning
        and sizing on the screen.

        Args:
            w1 (Widget): First widget for comparison.
            w2 (Widget): Second widget for comparison.

        Returns:
            bool: True if the IoU is greater than 0.9, False otherwise.
        """
        xa, ya = max(w1.bbox[0], w2.bbox[0]), max(w1.bbox[1], w2.bbox[1])
        xb, yb = min(w1.bbox[2], w2.bbox[2]), min(w1.bbox[3], w2.bbox[3])
        intersection = abs(max((xb - xa, 0)) * max((yb - ya), 0))
        boxa = abs((w1.bbox[2] - w1.bbox[0]) * (w1.bbox[3] - w1.bbox[1]))
        boxb = abs((w2.bbox[2] - w2.bbox[0]) * (w2.bbox[3] - w2.bbox[1]))
        iou = intersection / (boxa + boxb - intersection)
        return iou > 0.9
