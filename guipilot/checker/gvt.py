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
    def check_widget_pair(
        self, w1: Widget, w2: Widget, wi1: np.ndarray, wi2: np.ndarray
    ) -> list[tuple]:
        def get_quantized_colors(image: np.ndarray, k=3) -> list[tuple[int, int, int]]:
            """Extract k main colors from the image"""
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
            """Calculates redmean color distance, weight adjusted for human perception"""
            r1, g1, b1 = color1
            r2, g2, b2 = color2
            max_dist = 764.8339663572415
            mean_r = (r1 + r2) / 2
            delta_r, delta_g, delta_b = r1 - r2, g1 - g2, b1 - b2
            weight_r, weight_g, weight_b = 2 + mean_r / 256, 4, 2 + (255 - mean_r) / 256
            dist = math.sqrt(
                weight_r * delta_r**2
                + weight_g * delta_g**2
                + weight_b * delta_b**2
            )
            return dist / max_dist

        def check_bbox_consistency(w1: Widget, w2: Widget) -> bool:
            """Check if both widgets have similar position, size, and shape on the screen"""
            xa, ya = max(w1.bbox[0], w2.bbox[0]), max(w1.bbox[1], w2.bbox[1])
            xb, yb = min(w1.bbox[2], w2.bbox[2]), min(w1.bbox[3], w2.bbox[3])
            intersection = abs(max((xb - xa, 0)) * max((yb - ya), 0))
            boxa = abs((w1.bbox[2] - w1.bbox[0]) * (w1.bbox[3] - w1.bbox[1]))
            boxb = abs((w2.bbox[2] - w2.bbox[0]) * (w2.bbox[3] - w2.bbox[1]))
            iou = intersection / (boxa + boxb - intersection)
            return iou > 0.9

        def check_color_consistency(wi1: np.ndarray, wi2: np.ndarray) -> bool:
            """Perform color quantization to get color histograms, compare redmean distance between color pairs from histograms"""
            colors1 = get_quantized_colors(wi1, 3)
            colors2 = get_quantized_colors(wi2, 3)
            return all(
                [
                    get_color_distance(color1, color2) <= 0.01
                    for color1, color2 in zip(colors1, colors2)
                ]
            )

        def check_pid_consistency(wi1: np.ndarray, wi2: np.ndarray) -> bool:
            """Perform binary thresholding and use perceptual image differencing on binarized images"""
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
