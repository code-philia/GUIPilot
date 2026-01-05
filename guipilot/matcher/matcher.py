from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import TypeAlias

from guipilot.entities import Bbox

if typing.TYPE_CHECKING:
    from guipilot.entities import Screen, Widget


Pair: TypeAlias = tuple[int, int]
Score: TypeAlias = float


class WidgetMatcher(ABC):
    """
    Abstract base class for widget matching algorithms.

    This class defines the standard interface for algorithms that match widgets
    across two different GUI screens. It also provides utility methods for
    coordinate normalization to facilitate spatial comparisons.
    """

    @abstractmethod
    def match(
        self, screen_i: Screen, screen_j: Screen
    ) -> tuple[list[Pair], list[Score], float]:
        """
        Matches widgets between two provided screens.

        Subclasses must implement this method to define specific matching logic,
        such as heuristic alignment or vision-language model-based inference.

        Args:
            screen_i (Screen): The source screen containing a list of widgets to match.
            screen_j (Screen): The target screen containing potential matching widgets.

        Returns:
            tuple[list[Pair], list[Score], float]: A tuple containing:
                - list[Pair]: A list of tuples `(x, y)`, where `x` is a widget ID from
                  `screen_i` and `y` is a matching widget ID from `screen_j`.
                - list[Score]: A list of confidence or similarity scores for each pair.
                - float: The total execution time for the matching process in seconds.
        """
        pass

    def _norm_xywh(
        self, screen: Screen, widget: Widget
    ) -> tuple[float, float, float, float]:
        """
        Calculates the normalized bounding box of a widget in (x, y, w, h) format.

        The coordinates and dimensions are normalized relative to the screen width
        and height, resulting in values within the range [0, 1].

        Args:
            screen (Screen): The screen object used to provide image dimensions.
            widget (Widget): The widget whose bounding box is to be normalized.

        Returns:
            tuple[float, float, float, float]: A tuple containing normalized values:
                (xmin_norm, ymin_norm, width_norm, height_norm).

        Raises:
            AssertionError: If the screen is not in portrait orientation (height <= width)
                or if the calculated normalized coordinates fall outside the [0, 1] range.
        """
        screen_height, screen_width, _ = screen.image.shape
        xmin, ymin, xmax, ymax = widget.bbox
        xmin, xmax = xmin / screen_width, xmax / screen_width
        ymin, ymax = ymin / screen_height, ymax / screen_height

        # Validation checks for coordinate integrity
        assert screen_height > screen_width
        assert 0 <= xmin <= 1 and 0 <= xmax <= 1
        assert 0 <= ymin <= 1 and 0 <= ymax <= 1
        assert xmin <= xmax and ymin <= ymax

        return xmin, ymin, xmax - xmin, ymax - ymin
