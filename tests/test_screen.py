from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from guipilot.entities.constants import Bbox
from guipilot.entities.screen import Screen
from guipilot.entities.widget import Widget, WidgetType


def test_screen_init():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    screen = Screen(image=image)
    assert screen.image.shape == (100, 200, 3)
    assert screen.widgets == {}


@patch("guipilot.entities.screen.detector")
def test_screen_detect(mock_detector):
    # Mock detector returns bboxes and types
    mock_bboxes = [np.array([10, 10, 50, 50])]
    mock_types = ["textbutton"]
    mock_detector.return_value = (mock_bboxes, mock_types)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    screen = Screen(image=image)
    screen.detect()

    assert len(screen.widgets) == 1
    assert screen.widgets[0].type == WidgetType.TEXT_BUTTON
    assert screen.widgets[0].bbox == Bbox(10, 10, 50, 50)


@patch("guipilot.entities.screen.ocr")
def test_screen_ocr(mock_ocr):
    # Mock OCR returns texts and text bboxes
    mock_ocr.return_value = (["Click Me"], [Bbox(5, 5, 45, 45)])

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    widget = Widget(type=WidgetType.TEXT_BUTTON, bbox=Bbox(10, 10, 60, 60))
    screen = Screen(image=image, widgets={0: widget})

    screen.ocr()

    assert widget.texts == ["Click Me"]
    assert len(widget.text_bboxes) == 1
