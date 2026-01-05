from __future__ import annotations

import typing
from timeit import default_timer as timer

import numpy as np
from sklearn.neighbors import NearestNeighbors

from guipilot.matcher import Pair, Score, WidgetMatcher

if typing.TYPE_CHECKING:
    from guipilot.entities import Screen


class GVT(WidgetMatcher):
    """
    Graphical Verification Tool (GVT) widget matcher.

    This class implements a spatial-based matching algorithm that pairs widgets
    between two screens by calculating the Manhattan distance between their
    normalized bounding box coordinates. It uses a k-Nearest Neighbors (k-NN)
    approach to identify the closest candidates.
    """

    def __init__(self, threshold: float) -> None:
        """
        Initializes the GVT matcher with a distance threshold.

        Args:
            threshold (float): The maximum allowable Manhattan distance for a
                pair of widgets to be considered a match. Pairs with a distance
                exceeding this value are discarded.
        """
        super().__init__()
        self.threshold = threshold

    def match(
        self, screen_i: Screen, screen_j: Screen
    ) -> tuple[list[Pair], list[Score], float]:
        """
        Matches widgets between two screens based on spatial proximity.

        The method normalizes widget coordinates to a 0.0-1.0 scale and employs
        a k-NN search to find the nearest spatial neighbor for each widget in the
        source screen. Matches are established greedily, starting with the
        shortest distances.

        Args:
            screen_i (Screen): The source screen (e.g., mockup) containing widgets.
            screen_j (Screen): The target screen (e.g., implementation) to match against.

        Returns:
            tuple[list[Pair], list[Score], float]: A tuple containing:
                - list[Pair]: A list of tuples `(id_i, id_j)` representing
                  successfully matched widget IDs.
                - list[Score]: A list of similarity scores, calculated as the
                  reciprocal of the Manhattan distance (1/distance).
                - float: The total execution time of the matching process in
                  milliseconds.
        """
        start_time = timer()
        widget_keys_i, widget_keys_j = list(screen_i.widgets.keys()), list(
            screen_j.widgets.keys()
        )

        # Extract and normalize coordinates (x, y, w, h) for all widgets
        points_i = np.array(
            [
                list(self._norm_xywh(screen_i, widget))
                for widget in screen_i.widgets.values()
            ]
        )
        points_j = np.array(
            [
                list(self._norm_xywh(screen_j, widget))
                for widget in screen_j.widgets.values()
            ]
        )

        # Fit k-NN model on target screen coordinates using Manhattan metric
        knn = NearestNeighbors(n_neighbors=1, metric="manhattan")
        knn.fit(points_j)
        distances, indices = knn.kneighbors(points_i)

        # Sort potential matches by distance to prioritize the closest pairs
        sorted_distances_indices = sorted(
            enumerate(zip(distances, indices)),
            key=lambda x: x[1][0][0],  # Sort by the distance value
        )

        paired_ids = set()
        pairs, scores = [], []
        for i, (distance, index) in sorted_distances_indices:
            # Check if distance is within the user-defined threshold
            if distance[0] <= self.threshold:
                widget_i = widget_keys_i[i]
                widget_j = widget_keys_j[int(index[0])]

                # Skip if either widget has already been assigned to a pair (one-to-one matching)
                if widget_i in paired_ids or widget_j in paired_ids:
                    continue

                # Record the valid pair and calculate the confidence score
                pairs.append((widget_i, widget_j))
                scores.append(1 / distance[0] if distance[0] > 0 else 1.0)

                # Mark these IDs as occupied to maintain one-to-one mapping
                paired_ids.add(widget_i)
                paired_ids.add(widget_j)

        # Calculate time taken and convert to integer milliseconds
        time = (timer() - start_time) * 1000
        return pairs, scores, int(time)
