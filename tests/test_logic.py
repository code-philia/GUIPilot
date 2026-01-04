import numpy as np

from guipilot.checker.gvt import GVT as GVTChecker
from guipilot.entities.constants import Bbox, Inconsistency
from guipilot.entities.screen import Screen
from guipilot.entities.widget import Widget, WidgetType
from guipilot.matcher.gvt import GVT as GVTMatcher


def test_gvt_matcher_logic():
    # Setup two screens with similar widgets
    image1 = np.zeros((1000, 500, 3), dtype=np.uint8)
    image2 = np.zeros((1000, 500, 3), dtype=np.uint8)

    # Widget in screen1 at (100, 100)
    w1 = Widget(type=WidgetType.TEXT_BUTTON, bbox=Bbox(100, 100, 200, 150))
    # Widget in screen2 slightly shifted at (105, 105)
    w2 = Widget(type=WidgetType.TEXT_BUTTON, bbox=Bbox(105, 105, 205, 155))

    screen1 = Screen(image=image1, widgets={0: w1})
    screen2 = Screen(image=image2, widgets={0: w2})

    matcher = GVTMatcher(threshold=0.1)
    pairs, scores, time_taken = matcher.match(screen1, screen2)

    assert len(pairs) == 1
    assert pairs[0] == (0, 0)
    assert time_taken >= 0


def test_gvt_checker_logic():
    image1 = np.zeros((100, 100, 3), dtype=np.uint8)
    image2 = np.zeros((100, 100, 3), dtype=np.uint8)

    # Same widgets
    w1 = Widget(type=WidgetType.TEXT_BUTTON, bbox=Bbox(10, 10, 50, 50))
    w2 = Widget(type=WidgetType.TEXT_BUTTON, bbox=Bbox(10, 10, 50, 50))

    screen1 = Screen(image=image1, widgets={0: w1})
    screen2 = Screen(image=image2, widgets={0: w2})

    checker = GVTChecker()
    # Mocking pairs: widget 0 in screen1 matches widget 0 in screen2
    results, time_taken = checker.check(screen1, screen2, [(0, 0)])

    # Should be consistent
    assert len(results) == 0


def test_gvt_checker_inconsistency():
    image1 = np.zeros((100, 100, 3), dtype=np.uint8)
    image1[10:50, 10:50] = [255, 0, 0]  # Red

    image2 = np.zeros((100, 100, 3), dtype=np.uint8)
    image2[10:50, 10:50] = [0, 255, 0]  # Green

    w1 = Widget(type=WidgetType.TEXT_BUTTON, bbox=Bbox(10, 10, 50, 50))
    w2 = Widget(type=WidgetType.TEXT_BUTTON, bbox=Bbox(10, 10, 50, 50))

    screen1 = Screen(image=image1, widgets={0: w1})
    screen2 = Screen(image=image2, widgets={0: w2})

    checker = GVTChecker()
    results, _ = checker.check(screen1, screen2, [(0, 0)])

    # Should find color inconsistency
    # Note: GVT checker returns (x, y, inconsistency_type)
    assert any(res[2] == Inconsistency.COLOR for res in results if len(res) == 3)
