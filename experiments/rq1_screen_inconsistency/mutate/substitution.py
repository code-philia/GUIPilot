import random
import string
from copy import deepcopy

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from guipilot.entities import Bbox, Inconsistency, Screen, Widget, WidgetType

from .utils import get_context_color, sample_p


def swap_widgets(screen: Screen, p: float) -> tuple[Screen, set]:
    """Swap the positions of pairs of widgets that are of different types"""

    def get_area_score(i: Widget, j: Widget) -> float:
        """Calculate the ratio of widths of widgets."""
        areas = [i.area, j.area]
        return min(areas) / max(areas)

    def get_width_score(i: Widget, j: Widget) -> float:
        """Calculate the ratio of widths of widgets."""
        widths = [i.width, j.width]
        return min(widths) / max(widths)

    def get_height_score(i: Widget, j: Widget) -> float:
        """Calculate the ratio of heights of widgets."""
        heights = [i.height, j.height]
        return min(heights) / max(heights)

    def swap(screen: Screen, i: Widget, j: Widget) -> None:
        """Swap widgets on the screen, update their information"""
        # Extract the patches
        xmin_i, ymin_i, xmax_i, ymax_i = i.bbox
        xmin_j, ymin_j, xmax_j, ymax_j = j.bbox
        image_i = screen.image[ymin_i:ymax_i, xmin_i:xmax_i].copy()
        image_j = screen.image[ymin_j:ymax_j, xmin_j:xmax_j].copy()

        # Fill in previous patch locations with whitespace
        screen.image[ymin_i:ymax_i, xmin_i:xmax_i] = (255, 255, 255)
        screen.image[ymin_j:ymax_j, xmin_j:xmax_j] = (255, 255, 255)

        # Determine the destination bounds for each patch
        h, w, _ = screen.image.shape
        point_a, point_b = i.bbox[:2], j.bbox[:2]
        bbox_a = Bbox(
            point_a[0],
            point_a[1],
            min(point_a[0] + j.width, w),
            min(point_a[1] + j.height, h),
        )
        bbox_b = Bbox(
            point_b[0],
            point_b[1],
            min(point_b[0] + i.width, w),
            min(point_b[1] + i.height, h),
        )

        # Update widget bboxes
        i.bbox = bbox_b
        j.bbox = bbox_a

        # Perform the swap with adjusted dimensions
        screen.image[bbox_a[1] : bbox_a[3], bbox_a[0] : bbox_a[2]] = image_j[
            : j.height, : j.width
        ]
        screen.image[bbox_b[1] : bbox_b[3], bbox_b[0] : bbox_b[2]] = image_i[
            : i.height, : i.width
        ]

    screen = deepcopy(screen)
    n = len(screen.widgets)
    scores = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            widget_i, widget_j = screen.widgets[i], screen.widgets[j]
            if widget_i.type == widget_j.type:
                continue
            width_score = get_width_score(widget_i, widget_j)
            height_score = get_height_score(widget_i, widget_j)
            area_score = get_area_score(widget_i, widget_j)
            scores[i][j] = width_score * height_score * area_score

    row_indices, col_indices = linear_sum_assignment(scores, maximize=True)
    pairs = [(row, col) for row, col in zip(row_indices, col_indices)]
    pairs = sample_p(pairs, p)
    used = set()

    for pair in pairs:
        row, col = pair
        if row in used or col in used:
            continue
        widget_i = screen.widgets[row]
        widget_j = screen.widgets[col]
        try:
            swap(screen, widget_i, widget_j)
            used.add(row), used.add(col)
        except:
            continue

    # Extract widget data and their bounding boxes
    widgets = list(screen.widgets.values())
    widget_ids = list(screen.widgets.keys())
    bboxes = np.array([widget.bbox for widget in widgets])

    # Sort widgets and their ids based on bbox coordinates
    # Reassign sorted widgets and ids to screen
    sorted_indices = np.lexsort((bboxes[:, 0], bboxes[:, 1]))
    sorted_widgets = [widgets[i] for i in sorted_indices]
    sorted_widget_ids = [widget_ids[i] for i in sorted_indices]
    screen.widgets = dict(zip(sorted_widget_ids, sorted_widgets))

    changed = {(id1, id2, Inconsistency.BBOX) for id1, id2 in pairs if id1 != id2} | {
        (id2, id1, Inconsistency.BBOX) for id1, id2 in pairs if id1 != id2
    }
    return screen, changed


def change_widgets_text(screen: Screen, p: float) -> tuple[Screen, set]:
    """Change the text of text-based widgets"""

    def get_random_text(length: int) -> str:
        """Generate a random string of length."""
        letters = string.ascii_letters
        return "".join(random.choice(letters) for _ in range(length))

    def get_max_font_scale(text: str, bbox: list, font: int, thickness: int) -> float:
        """Calculate the maximum font scale such that the text fits inside the bbox."""
        xmin, ymin, xmax, ymax = bbox
        box_width = xmax - xmin
        box_height = ymax - ymin
        font_scale, max_scale = 1, 10
        while font_scale < max_scale:
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_width, text_height = text_size

            if text_width <= box_width and text_height <= box_height:
                font_scale += 0.1
            else:
                break

        return font_scale - 0.1

    screen = deepcopy(screen)
    text_based_widgets = {WidgetType.TEXT_VIEW, WidgetType.TEXT_BUTTON}
    widgets = {
        id: widget
        for id, widget in screen.widgets.items()
        if widget.type in text_based_widgets
    }
    widgets: dict[int, Widget] = sample_p(widgets, p)
    changed = set()

    for id, widget in widgets.items():
        if not widget.texts or not widget.text_bboxes:
            continue

        for text, bbox in zip(widget.texts, widget.text_bboxes):
            # Replace with random text that scales within the bbox
            text = get_random_text(len(text))
            xmin, ymin, xmax, ymax = map(int, bbox)
            rel_xmin, rel_ymin, _, _ = map(int, widget.bbox)
            xmin = rel_xmin + xmin
            ymin = rel_ymin + ymin
            xmax = rel_xmin + xmax
            ymax = rel_ymin + ymax

            font, thickness = cv2.FONT_HERSHEY_SIMPLEX, 2
            text_image = screen.image[ymin:ymax, xmin:xmax]
            font_scale = get_max_font_scale(
                text, (xmin, ymin, xmax, ymax), font, thickness
            )

            # Fill bbox with widget context color and insert text
            bg_color = get_context_color(text_image)
            brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

            screen.image[ymin:ymax, xmin:xmax] = bg_color
            screen.image = cv2.putText(
                screen.image,
                text,
                (xmin, ymax),
                font,
                font_scale,
                text_color,
                thickness,
            )

        changed.add((id, id, Inconsistency.TEXT))

    return screen, changed


def change_widgets_color(screen: Screen, p: float) -> tuple[Screen, set]:
    """Change the color of image-based widgets"""

    def transform(image: np.ndarray) -> np.ndarray:
        # Apply bilateral filter to reduce noise while preserving edges
        filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        # Define a broader mask for the whitespace areas
        hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)
        sensitivity = 15
        lower_white = np.array([0, 0, 255 - sensitivity])
        upper_white = np.array([255, sensitivity, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        non_whitespace_mask = cv2.bitwise_not(mask)

        # Generate a random color
        random_color = [random.randint(0, 255) for _ in range(3)]
        colored_image = np.full(image.shape, random_color, dtype=np.uint8)

        # Use the mask to change the color of non-whitespace areas
        result_image = cv2.bitwise_and(
            colored_image, colored_image, mask=non_whitespace_mask
        )
        masked_whitespace = cv2.bitwise_and(filtered_image, filtered_image, mask=mask)
        result_image = cv2.add(result_image, masked_whitespace)
        return result_image

    screen = deepcopy(screen)
    image_based_widgets = {
        WidgetType.ICON_BUTTON,
        WidgetType.COMBINED_BUTTON,
        WidgetType.IMAGE_VIEW,
        WidgetType.CHART,
    }
    widgets = {
        id: widget
        for id, widget in screen.widgets.items()
        if widget.type in image_based_widgets
    }
    widgets = sample_p(widgets, p)
    changed = set()

    for id, widget in widgets.items():
        xmin, ymin, xmax, ymax = widget.bbox
        widget_image = screen.image[ymin:ymax, xmin:xmax]
        widget_image = transform(widget_image)
        screen.image[ymin:ymax, xmin:xmax] = widget_image
        changed.add((id, id, Inconsistency.COLOR))

    return screen, changed
