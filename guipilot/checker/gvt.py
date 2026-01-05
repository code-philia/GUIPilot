from __future__ import annotations

import math
import re
import typing
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image

from guipilot.entities import Inconsistency, WidgetType

from .checker import ScreenChecker

if typing.TYPE_CHECKING:
    from guipilot.entities import Widget


class GVT(ScreenChecker):
    """GVT (Graphical Verification Tool) anomaly detector implementation class.

    This class inherits from ScreenChecker and adopts more refined visual verification methods, including
    perceptual weight-based color distance calculation and Perceptual Image Difference (PID) analysis,
    to detect advanced visual inconsistencies between design mockups and implementations.
    """

    def check_widget_pair(
        self, w1: Widget, w2: Widget, wi1: np.ndarray, wi2: np.ndarray
    ) -> list[tuple]:
        """Evaluate visual and content consistency between a pair of matched widgets.

        This method comprehensively performs in-depth checks based on spatial position, text content,
        quantized color distribution, and pixel-level perceptual differences.

        Args:
            w1 (Widget): Widget entity from the first screen (e.g., design mockup).
            w2 (Widget): Widget entity from the second screen (e.g., actual application).
            wi1 (np.ndarray): Cropped image region of widget w1 (BGR format).
            wi2 (np.ndarray): Cropped image region of widget w2 (BGR format).

        Returns:
            list[tuple]: List containing detected inconsistency types (Inconsistency enum values).
        """

        def get_quantized_colors(image: np.ndarray, k=3) -> list[tuple[int, int, int]]:
            """Extract k dominant colors from an image.

            Uses Median Cut algorithm for image quantization to compress continuous color space
            into discrete dominant colors.

            Args:
                image (np.ndarray): Image array in BGR format.
                k (int, optional): Number of dominant colors to extract. Defaults to 3.

            Returns:
                list[tuple[int, int, int]]: List of RGB color tuples. Returns white by default if colors are invalid.
            """
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image: Image.Image = Image.fromarray(image)
            image = image.quantize(
                colors=3, method=Image.Quantize.MEDIANCUT, dither=Image.NONE, kmeans=0
            )
            palette = image.getpalette()
            colors_list: list[int] = [palette[i : i + 3] for i in range(0, k * 3, 3)]
            return [[255, 255, 255] if not color else color for color in colors_list]

        def get_color_distance(
            color1: tuple[int, int, int], color2: tuple[int, int, int]
        ) -> float:
            """Calculate Redmean distance between two colors.

            The Redmean algorithm applies weighted adjustments to RGB differences based on human visual
            sensitivity to colors of different wavelengths.

            Args:
                color1 (tuple): First RGB color tuple.
                color2 (tuple): Second RGB color tuple.

            Returns:
                float: Normalized color distance (between 0.0 and 1.0).
            """
            r1, g1, b1 = color1
            r2, g2, b2 = color2
            max_dist = 764.8339663572415
            mean_r = (r1 + r2) / 2
            delta_r, delta_g, delta_b = r1 - r2, g1 - g2, b1 - b2
            weight_r, weight_g, weight_b = 2 + mean_r / 256, 4, 2 + (255 - mean_r) / 256
            dist = math.sqrt(
                weight_r * delta_r**2 + weight_g * delta_g**2 + weight_b * delta_b**2
            )
            return dist / max_dist

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

        def check_color_consistency(wi1: np.ndarray, wi2: np.ndarray) -> bool:
            """Check color histogram consistency between widgets.

            Verifies if the perceptual distance of extracted dominant color pairs between two widgets
            is within a minimal threshold.

            Args:
                wi1 (np.ndarray): First widget image.
                wi2 (np.ndarray): Second widget image.

            Returns:
                bool: Returns True if the distance of all corresponding colors is <= 0.01.
            """
            colors1 = get_quantized_colors(wi1, 3)
            colors2 = get_quantized_colors(wi2, 3)
            return all(
                get_color_distance(color1, color2) <= 0.01
                for color1, color2 in zip(colors1, colors2)
            )

        def check_pid_consistency(wi1: np.ndarray, wi2: np.ndarray) -> bool:
            """Perform Perceptual Image Differencing (PID) check.

            Detects visually perceptible changes in images through binarization thresholding
            and differential pixel ratio analysis.

            Args:
                wi1 (np.ndarray): First widget image.
                wi2 (np.ndarray): Second widget image.

            Returns:
                bool: Considered consistent and returns True if differential pixel ratio is below 20%.
            """
            h1, w1, _ = wi1.shape
            h2, w2, _ = wi2.shape
            h3, w3 = max(h1, h2), max(w1, w2)
            wi1 = cv2.cvtColor(wi1, cv2.COLOR_BGR2GRAY)
            wi1 = cv2.resize(wi1, (w3, h3), interpolation=cv2.INTER_AREA)
            wi2 = cv2.cvtColor(wi2, cv2.COLOR_BGR2GRAY)
            wi2 = cv2.resize(wi2, (w3, h3), interpolation=cv2.INTER_AREA)
            absdiff = cv2.absdiff(wi1, wi2)
            _, thresholded = cv2.threshold(
                absdiff, int(0.1 * 255), 255, cv2.THRESH_BINARY
            )
            diff_ratio = np.count_nonzero(thresholded) / (h3 * w3)
            return diff_ratio <= 0.2

        def check_text_consistency(w1: Widget, w2: Widget) -> bool:
            """Check text content consistency between widgets.

            Args:
                w1 (Widget): First widget.
                w2 (Widget): Second widget.

            Returns:
                bool: Returns True if text matching ratio >= 0.95.
            """
            return super().check_text_consistency(w1, w2)

        diff = set()
        if not check_bbox_consistency(w1, w2):
            diff.add(Inconsistency.BBOX)
        if not check_text_consistency(w1, w2):
            diff.add(Inconsistency.TEXT)
        if not check_color_consistency(wi1, wi2):
            diff.add(Inconsistency.COLOR)
        if not check_pid_consistency(wi1, wi2):
            diff.add(Inconsistency.COLOR)

        return list(diff)
