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
    """
    Consistency checker for mobile GUI elements.

    This class implements the core logic for detecting discrepancies between
    paired widgets from different screens (e.g., a mockup design and its
    actual implementation) across multiple dimensions including layout,
    content, and appearance.
    """

    def check_widget_pair(
        self, w1: Widget, w2: Widget, wi1: np.ndarray, wi2: np.ndarray
    ) -> list[tuple]:
        """
        Detects inconsistencies between a matched pair of widgets.

        This method orchestrates individual checks for bounding box alignment,
        textual similarity, and visual color distribution to identify
        specific types of implementation errors.

        Args:
            w1 (Widget): The widget entity from the source screen (e.g., mockup).
            w2 (Widget): The widget entity from the target screen (e.g., implementation).
            wi1 (np.ndarray): The cropped image data of widget w1.
            wi2 (np.ndarray): The cropped image data of widget w2.

        Returns:
            list[Inconsistency]: A list of detected inconsistency types (BBOX, TEXT, or COLOR).
        """

        def check_bbox_consistency(w1: Widget, w2: Widget) -> bool:
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
            return super().check_bbox_consistency(w1, w2)

        def check_text_consistency(w1: Widget, w2: Widget) -> bool:
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
            return super().check_text_consistency(w1, w2)

        def check_color_consistency(wi1: np.ndarray, wi2: np.ndarray) -> bool:
            """
            Evaluates the visual color similarity between two widget images.

            Computes normalized 3D color histograms (8 bins per RGB channel) for both
            images and calculates the distance using Kullback-Leibler (KL) Divergence.
            A lower score indicates more similar color distributions.

            Args:
                wi1 (np.ndarray): Image array of the first widget.
                wi2 (np.ndarray): Image array of the second widget.

            Returns:
                bool: True if the KL Divergence score is less than 8.
            """
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
