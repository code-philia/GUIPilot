from __future__ import annotations

import typing
from timeit import default_timer as timer

import numpy as np

from .matcher import Pair, Score, WidgetMatcher

if typing.TYPE_CHECKING:
    from guipilot.entities import Screen, Widget


class GUIPilotV2(WidgetMatcher):
    def __init__(self, s1: int = 100, s2: int = 1) -> None:
        """
        Params
            s1: scaling factor for distance score i.e. s1 * (abs(xi - xj) + abs(yi - yj))
            s2: scaling factor for shape score i.e. s2 * (abs(wi - wj) + abs(hi - hj))
        """
        self.s1, self.s2 = s1, s2
        super().__init__()

    def match(
        self, screen_i: Screen, screen_j: Screen
    ) -> tuple[list[Pair], list[Score], float]:
        start_time = timer()
        widget_keys_i, widget_keys_j = list(screen_i.widgets.keys()), list(
            screen_j.widgets.keys()
        )
        scores = self._calculate_match_scores(screen_i, screen_j)
        path = self._find_longest_matching_subsequence(scores)
        pairs = [(widget_keys_i[x], widget_keys_j[y]) for x, y in path]
        scores = [scores[x, y] for (x, y) in path]
        time = (timer() - start_time) * 1000
        return pairs, scores, int(time)

    def _calculate_match_scores(self, screen_i: Screen, screen_j: Screen) -> np.ndarray:
        def get_distance_score(si: Screen, sj: Screen, i: Widget, j: Widget) -> float:
            """Calculates Manhattan distance between normalized bboxes of widgets."""
            si_h, si_w, _ = si.image.shape
            sj_h, sj_w, _ = sj.image.shape

            xi, yi, wi, hi = (
                i.bbox.xmin / si_w,
                i.bbox.ymin / si_h,
                i.width / si_w,
                i.height / si_h,
            )
            xj, yj, wj, hj = (
                j.bbox.xmin / sj_w,
                j.bbox.ymin / sj_h,
                j.width / sj_w,
                j.height / sj_h,
            )

            score = self.s1 * (abs(xi - xj) + abs(yi - yj)) + self.s2 * (
                abs(wi - wj) + abs(hi - hj)
            )
            return min(1 / score, 1) if score else 1

        def get_area_score(i: Widget, j: Widget) -> float:
            """Calculates the ratio of widget areas."""
            areas = [i.area, j.area]
            return min(areas) / max(areas)

        def get_shape_score(i: Widget, j: Widget) -> float:
            """Calculate the ratio of aspect ratios of widgets."""
            aspect_ratios = [i.width / i.height, j.width / j.height]
            return min(aspect_ratios) / max(aspect_ratios)

        def get_type_score(i: Widget, j: Widget) -> float:
            """Compare the type compatibility of widgets."""
            if i.type == j.type:
                return 1
            else:
                return 0.01

        widgets_i, widgets_j = screen_i.widgets.values(), screen_j.widgets.values()
        scores = np.zeros((len(widgets_i), len(widgets_j)))

        for i, widget_i in enumerate(widgets_i):
            for j, widget_j in enumerate(widgets_j):
                distance_score = get_distance_score(
                    screen_i, screen_j, widget_i, widget_j
                )
                area_score = get_area_score(widget_i, widget_j)
                shape_score = get_shape_score(widget_i, widget_j)
                type_score = get_type_score(widget_i, widget_j)

                score = distance_score * area_score * shape_score * type_score
                scores[i, j] = max(score, 1e-8)

        return scores

    def _find_longest_matching_subsequence(self, D: np.ndarray) -> list[tuple]:
        m, n = D.shape
        if m == 0 or n == 0:
            return []
        dp = np.zeros((m + 1, n + 1))
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i, j] = max(
                    dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1] + D[i - 1, j - 1]
                )
        i, j = m, n
        sequence = []
        while i > 0 and j > 0:
            if dp[i, j] == dp[i - 1, j - 1] + D[i - 1, j - 1]:
                sequence.append((i - 1, j - 1))
                i, j = i - 1, j - 1
            elif dp[i, j] == dp[i - 1, j]:
                i -= 1
            else:
                j -= 1
        return sequence[::-1]
