from __future__ import annotations

import typing
from timeit import default_timer as timer

if typing.TYPE_CHECKING:
    from guipilot.checker import ScreenChecker
    from guipilot.matcher import WidgetMatcher

    from .screen import Screen


class Process(object):
    def __init__(self) -> None:
        self.screens: list[Screen] = []

    def add(self, screen: Screen) -> None:
        """Add a screen to the end of process"""
        self.screens.append(screen)

    def check(
        self,
        target: Screen,
        matcher: WidgetMatcher,
        checker: ScreenChecker,
        process_path,
        i,
    ) -> tuple[bool, float]:
        """Check for process inconsistency on the current screen

        The current screen is the most recent screen the process is on. This method compares the
        target screen against all screens in the process. If the target screen has the fewest
        inconsistencies with the current screen, the process is considered consistent so far.

        Args:
            target: The screen to check against
            matcher: An algorithm that pairs matching widgets on both screens.
            checker: An algorithm that checks the consistency of paired widgets.

        Returns:
            bool: True if the process is consistent, otherwise False.
        """
        start_time = timer()
        screen = self.screens[-1]
        pairs, scores = matcher.match(screen, target)
        inconsistencies, _ = checker.check(screen, target, pairs)

        unpaired_screen = set([x[0] for x in inconsistencies if x[1] is None])
        unpaired_screen = len(unpaired_screen) / len(screen.widgets)

        unpaired_target = set([x[1] for x in inconsistencies if x[0] is None])
        unpaired_target = len(unpaired_target) / len(target.widgets)

        matching_score = sum(scores) / len(target.widgets)

        a = matching_score
        b = matching_score - unpaired_screen
        c = matching_score - unpaired_target

        return (a, b, c), timer() - start_time
