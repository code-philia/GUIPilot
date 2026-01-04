from copy import deepcopy

import numpy as np

from guipilot.entities import Bbox, Inconsistency, Screen, Widget

from .utils import get_context_color, sample_p


def delete_widgets(screen: Screen, p: float) -> tuple[Screen, set]:
    """Mask a widget by its context color"""
    screen = deepcopy(screen)
    widgets: dict[int, Widget] = sample_p(screen.widgets, p)
    remove = set()
    for id, widget in widgets.items():
        remove.add(id)
        xmin, ymin, xmax, ymax = widget.bbox
        widget_image = screen.image[ymin:ymax, xmin:xmax]
        color = get_context_color(widget_image)
        screen.image[ymin:ymax, xmin:xmax] = color

    screen.widgets = {
        id: widget for id, widget in screen.widgets.items() if id not in remove
    }
    changed = set([(id, None) for id in remove])
    return screen, changed


def delete_row(screen: Screen, p: float) -> tuple[Screen, set]:
    """
    1. Select a random widget `a` to be deleted
    2. Find all other widgets `b` on the same row as `a`, then trim the entire row
    3. Update the bboxes of all widgets `c` below the row
               ┌───┐
    ---┌───┐---│-b-│----
       │ a │   └───┘
    ---└───┘------------
               ┌───┐
               │ c │
               └───┘
    """
    screen = deepcopy(screen)
    widgets: dict[int, Widget] = sample_p(screen.widgets, p)
    remove = set()
    shifted = set()
    for a, widget_a in widgets.items():
        if a in remove:
            continue

        remove.add(a)
        _, ymin_a, _, ymax_a = widget_a.bbox

        # find all widgets on the same row
        for b, widget_b in screen.widgets.items():
            if b in remove:
                continue
            _, ymin_b, _, ymax_b = widget_b.bbox
            if not (ymax_b < ymin_a or ymin_b > ymax_a):
                remove.add(b)

        # trim entire row
        rows_to_remove = list(range(ymin_a, ymax_a))
        screen.image = np.delete(screen.image, rows_to_remove, axis=0)

        # update the bboxes of widgets below the row
        y_offset = ymax_a - ymin_a
        for c, widget_c in screen.widgets.items():
            if c in remove:
                continue
            xmin_c, ymin_c, xmax_c, ymax_c = widget_c.bbox
            if ymin_c > ymax_a:
                shifted.add(c)
                widget_c.bbox = Bbox(
                    xmin_c, ymin_c - y_offset, xmax_c, ymax_c - y_offset
                )

    screen.widgets = {
        id: widget for id, widget in screen.widgets.items() if id not in remove
    }
    changed = set(
        [(id, None) for id in remove] + [(id, id, Inconsistency.BBOX) for id in shifted]
    )
    return screen, changed
