from __future__ import annotations

import typing
from timeit import default_timer as timer

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .matcher import Pair, Score, WidgetMatcher

if typing.TYPE_CHECKING:
    from guipilot.entities import Screen


class GVT(WidgetMatcher):
    def __init__(self, threshold) -> None:
        super().__init__()
        self.threshold = threshold

    def match(
        self, screen_i: Screen, screen_j: Screen
    ) -> tuple[list[Pair], list[Score], float]:
        start_time = timer()
        widget_keys_i, widget_keys_j = list(screen_i.widgets.keys()), list(
            screen_j.widgets.keys()
        )
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

        knn = NearestNeighbors(n_neighbors=1, metric="manhattan")
        knn.fit(points_j)
        distances, indices = knn.kneighbors(points_i)
        sorted_distances_indices = sorted(
            enumerate(zip(distances, indices)),
            key=lambda x: x[1][0][0],  # Sort by the distance
        )

        paired_ids = set()
        pairs, scores = [], []
        for i, (distance, index) in sorted_distances_indices:
            if distance[0] <= self.threshold:
                widget_i = widget_keys_i[i]
                widget_j = widget_keys_j[int(index[0])]

                # Skip if either widget has already been paired
                if widget_i in paired_ids or widget_j in paired_ids:
                    continue

                # Add pair to the list
                pairs.append((widget_i, widget_j))
                scores.append(1 / distance[0])

                # Mark widgets as paired
                paired_ids.add(widget_i)
                paired_ids.add(widget_j)

        time = (timer() - start_time) * 1000
        return pairs, scores, int(time)
