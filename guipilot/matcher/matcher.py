from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import TypeAlias

if typing.TYPE_CHECKING:
    from guipilot.entities import Screen, Widget


Pair: TypeAlias = tuple[int, int]
Score: TypeAlias = float


class WidgetMatcher(ABC):
    @abstractmethod
    def match(
        self, screen_i: Screen, screen_j: Screen
    ) -> tuple[list[Pair], list[Score], float]:
        """Match widgets between two screens

        Args:
            screen_i, screen_j: two screens containing a list of widgets to match

        Returns:
            A list of tuples, where each tuple `(x, y)` represents a pair of matching
            widget IDs. `x` is from `screen_i` and `y` is from `screen_j`.
        """

    def _norm_xywh(
        self, screen: Screen, widget: Widget
    ) -> tuple[float, float, float, float]:
        """Calculate the normalized bounding box of a widget"""
        screen_height, screen_width, _ = screen.image.shape
        xmin, ymin, xmax, ymax = widget.bbox
        xmin, xmax = xmin / screen_width, xmax / screen_width
        ymin, ymax = ymin / screen_height, ymax / screen_height
        assert screen_height > screen_width
        assert 0 <= xmin <= 1 and 0 <= xmax <= 1
        assert 0 <= ymin <= 1 and 0 <= ymax <= 1
        assert xmin <= xmax and ymin <= ymax
        return xmin, ymin, xmax - xmin, ymax - ymin
