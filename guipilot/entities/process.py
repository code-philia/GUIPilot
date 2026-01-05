from __future__ import annotations

import typing
from timeit import default_timer as timer

if typing.TYPE_CHECKING:
    from guipilot.checker import ScreenChecker
    from guipilot.matcher import WidgetMatcher

    from .screen import Screen


class Process(object):
    """
    Represents a sequence of GUI transitions (a process flow).

    This class maintains a chronological list of screens representing an expected
    UI process (typically from mockups) and provides methods to validate real-world
    screens against this sequence.

    Attributes:
        screens (list[Screen]): A list of Screen objects representing the
            intended steps of the process.
    """

    def __init__(self) -> None:
        """Initializes an empty process flow."""
        self.screens: list[Screen] = []

    def add(self, screen: Screen) -> None:
        """
        Appends a screen to the end of the process flow.

        Args:
            screen (Screen): The Screen object to be added as the next step
                in the process.
        """
        self.screens.append(screen)

    def check(
        self,
        target: Screen,
        matcher: WidgetMatcher,
        checker: ScreenChecker,
        process_path,
        i,
    ) -> tuple[tuple[float, float, float], float]:
        """
        Evaluates the current screen in the process for flow inconsistencies.

        This method compares the provided target screen (implementation) against the
        most recent screen in the process (mockup). It calculates three specific
        consistency metrics (a, b, c) based on matching scores and the ratio of
        unpaired widgets on both the source and target screens.

        Args:
            target (Screen): The actual screen from the application to be verified.
            matcher (WidgetMatcher): The algorithm used to align widgets between
                the expected screen and the target screen.
            checker (ScreenChecker): The algorithm used to identify
                inconsistencies within matched widget pairs.
            process_path: The filesystem path or identifier for the process data
                (used for context).
            i: The current step index within the process.

        Returns:
            tuple[tuple[float, float, float], float]: A tuple containing:
                - tuple (a, b, c): Consistency scores where:
                    - 'a' is the raw matching score normalized by target widgets.
                    - 'b' is the matching score adjusted for unpaired source widgets.
                    - 'c' is the matching score adjusted for unpaired target widgets.
                - float: The time taken to perform the check in seconds.
        """
        start_time = timer()
        screen = self.screens[-1]
        pairs, scores = matcher.match(screen, target)
        inconsistencies, _ = checker.check(screen, target, pairs)

        # Calculate the ratio of widgets in the expected screen that have no match
        unpaired_screen = set([x[0] for x in inconsistencies if x[1] is None])
        unpaired_screen = len(unpaired_screen) / len(screen.widgets)

        # Calculate the ratio of widgets in the target screen that have no match
        unpaired_target = set([x[1] for x in inconsistencies if x[0] is None])
        unpaired_target = len(unpaired_target) / len(target.widgets)

        # Calculate overall matching confidence normalized by the target widget count
        matching_score = sum(scores) / len(target.widgets)

        a = matching_score
        b = matching_score - unpaired_screen
        c = matching_score - unpaired_target

        return (a, b, c), timer() - start_time
