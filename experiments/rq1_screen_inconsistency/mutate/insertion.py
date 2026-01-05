import glob
import os
import random
from copy import deepcopy

import numpy as np
from dotenv import load_dotenv
from utils import load_screen

from guipilot.entities import Bbox, Inconsistency, Screen, Widget

from .utils import sample_p

load_dotenv()
DATASET_PATH = os.getenv("DATASET_PATH")
IMAGE_PATHS = glob.glob(os.path.join(DATASET_PATH, "*", "*", "*.jpg"))


def insert_widgets(screen: Screen, p: float) -> tuple[Screen, set]:
    """Add a widget to an empty space"""

    def get_maximal_rectangle(mask: list[list[int]]):
        """Find the largest rectangle among unoccupied areas in the mask"""
        if not mask:
            return 0, None, None

        rows, cols = len(mask), len(mask[0])
        heights = [0] * (cols + 1)
        max_area = 0
        max_bbox = (None, None, None, None)

        for r in range(rows):
            for i in range(cols):
                heights[i] = heights[i] + 1 if mask[r][i] == 1 else 0

            # Calculate max area using histogram method
            stack = []
            for i in range(len(heights)):
                while stack and heights[i] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    area = h * w

                    if area > max_area:
                        max_area = area
                        right = i - 1
                        left = right - w + 1
                        bottom = r
                        top = bottom - h + 1
                        max_bbox = (left, top, right, bottom)

                stack.append(i)

        return max_area, max_bbox

    screen = deepcopy(screen)
    height, width, _ = screen.image.shape

    # Select a random screen
    random_path = random.choice(IMAGE_PATHS)
    random_screen = load_screen(random_path)
    random_screen.ocr()

    # Sample some random widgets
    random_widgets: list[Widget] = list(random_screen.widgets.values())
    random_widgets.sort(key=lambda x: x.area)

    # Mask out regions occupied by widgets in the image
    unoccupied_mask = np.ones(shape=(height, width))
    for widget in screen.widgets.values():
        xmin, ymin, xmax, ymax = widget.bbox
        unoccupied_mask[ymin:ymax, xmin:xmax] = 0

    # Insert random widgets into unmasked regions
    new_widget_ids = []
    k = max(int(p * len(random_widgets)), 1)
    for i in range(k):
        _, max_bbox = get_maximal_rectangle(unoccupied_mask.tolist())

        widget = random_widgets[i]
        xmin1, ymin1, xmax1, ymax1 = widget.bbox
        xmin2, ymin2, _, _ = max_bbox

        image = random_screen.image[ymin1:ymax1, xmin1:xmax1].copy()

        xmax3, ymax3 = min(xmin2 + image.shape[1], width), min(
            ymin2 + image.shape[0], height
        )
        screen.image[ymin2:ymax3, xmin2:xmax3] = image[
            0 : ymax3 - ymin2, 0 : xmax3 - xmin2
        ]
        unoccupied_mask[
            ymin2 : ymin2 + image.shape[0], xmin2 : xmin2 + image.shape[1]
        ] = 0

        new_widget_id = max(screen.widgets.keys()) + 1
        new_widget_ids.append(new_widget_id)
        widget.bbox = Bbox(xmin2, ymin2, xmin2 + image.shape[1], ymin2 + image.shape[0])
        screen.widgets[new_widget_id] = widget

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

    changes = set({(None, id) for id in new_widget_ids})
    return screen, changes


def insert_row(screen: Screen, p: float) -> tuple[Screen, set]:
    r"""
        0. Pick a random screen
        1. Select a random widget `a` to be inserted
        2. Find all other widgets `b` on the same row as `a`, then calculate the total row height
        3. Insert the entire row into the screen
        4. Update the bboxes of all widgets `c` below the row

        -----------┌───┐---- \  
        ---┌───┐---│-b-│----  |_ inserted rows
           │ a │   └───┘      |
        ---└───┘------------ /
                   ┌───┐
                   │ c │
                   └───┘
    """
    screen = deepcopy(screen)

    # Select a random screen
    random_path = random.choice(IMAGE_PATHS)
    random_screen = load_screen(random_path)
    random_screen.ocr()

    # Sample some random widgets
    random_widgets: dict[int, Widget] = sample_p(random_screen.widgets, p)

    new_widget_ids = set()  # id relative to screen
    widget_ids = set()  # id relative to random screen
    shifted = set()
    for a, random_widget in random_widgets.items():
        selected = {a}

        if a in widget_ids:
            continue
        _, ymin_a, xmax_a, ymax_a = random_widget.bbox

        # find all widgets on the same row
        ymin_row, ymax_row, xmax_row = ymin_a, ymax_a, xmax_a
        for b, widget_b in random_screen.widgets.items():
            if b in widget_ids:
                continue
            _, ymin_b, xmax_b, ymax_b = widget_b.bbox

            if (
                not (ymax_b < ymin_a or ymin_b > ymax_a)
                and xmax_b <= screen.image.shape[1]
            ):
                selected.add(b)
                ymin_row = min(ymin_row, ymin_b)
                ymax_row = max(ymax_row, ymax_b)
                xmax_row = max(xmax_row, xmax_b)

        # Mask out regions occupied by widgets in the screen
        unoccupied_mask = np.ones(shape=(screen.image.shape[0], screen.image.shape[1]))
        for widget in screen.widgets.values():
            xmin, ymin, xmax, ymax = widget.bbox
            unoccupied_mask[ymin:ymax, xmin:xmax] = 0

        # Find rows that are unoccupied by any widgets
        unoccupied_rows = np.all(unoccupied_mask == 1, axis=1)
        unoccupied_rows = np.where(unoccupied_rows)[0]

        # Copy rows from random screen
        padding = 50
        new_rows = np.full(
            ((ymax_row - ymin_row) + padding * 2, screen.image.shape[1], 3),
            255,
            dtype=np.uint8,
        )
        new_width = min(screen.image.shape[1], xmax_row)
        new_rows[padding : padding + ymax_row - ymin_row, 0:new_width] = (
            random_screen.image[ymin_row:ymax_row, 0:new_width]
        )

        # Select an unoccupied row as insertion point, more likely to insert to middle of screen
        mean, std = (len(unoccupied_rows) - 1) / 2, len(unoccupied_rows) / 6
        weights = np.exp(-0.5 * ((np.arange(len(unoccupied_rows)) - mean) / std) ** 2)
        weights /= weights.sum()
        y_insertion = random.choices(unoccupied_rows, weights=weights, k=1)[0]
        screen.image = np.insert(screen.image, y_insertion, new_rows, axis=0)

        # update the bboxes of widgets below the inserted row
        y_offset = (ymax_row - ymin_row) + padding * 2

        for c, widget_c in screen.widgets.items():
            xmin_c, ymin_c, xmax_c, ymax_c = widget_c.bbox
            if ymax_c >= y_insertion:
                shifted.add(c)
                widget_c.bbox = Bbox(
                    xmin_c, ymin_c + y_offset, xmax_c, ymax_c + y_offset
                )

        # Add the new widgets to screen
        y_rel = min([random_screen.widgets[id].bbox.ymin for id in selected])
        for id in selected:
            new_widget = random_screen.widgets[id]
            xmin_new, ymin_new, xmax_new, ymax_new = new_widget.bbox
            if xmax_new > screen.image.shape[1]:
                continue
            new_widget_height = ymax_new - ymin_new

            new_widget.bbox = Bbox(
                xmin_new,
                y_insertion + padding + (ymin_new - y_rel),
                xmax_new,
                y_insertion + padding + (ymin_new - y_rel) + new_widget_height,
            )
            new_widget_id = max(screen.widgets.keys()) + 1
            new_widget_ids.add(new_widget_id)
            screen.widgets[new_widget_id] = new_widget
            widget_ids.add(id)

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

    changes = set(
        [(None, id) for id in new_widget_ids]
        + [(id, id, Inconsistency.BBOX) for id in shifted]
    )
    return screen, changes
