from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from timeit import default_timer as timer

import numpy as np

if typing.TYPE_CHECKING:
    from guipilot.entities import Screen, Widget


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
        unpaired_i = set([id for id in screen_i.widgets.keys()])
        unpaired_j = set([id for id in screen_j.widgets.keys()])

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
