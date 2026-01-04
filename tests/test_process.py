import numpy as np

from guipilot.entities.constants import Bbox
from guipilot.entities.process import Process
from guipilot.entities.screen import Screen
from guipilot.entities.widget import Widget, WidgetType


def test_add_screen():
    process = Process()
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    bbox = Bbox(10, 10, 50, 50)
    widget = Widget(type=WidgetType.TEXT_BUTTON, bbox=bbox)

    widgets = {0: widget}

    screen = Screen(image=image, widgets=widgets)

    process.add(screen)


def test_add_multiple_screens():
    process = Process()
    image1 = np.zeros((100, 100, 3), dtype=np.uint8)
    image2 = np.ones((100, 100, 3), dtype=np.uint8)

    screen1 = Screen(image=image1)
    screen2 = Screen(image=image2)

    process.add(screen1)
    process.add(screen2)

    assert len(process.screens) == 2
    assert process.screens[0] is screen1
    assert process.screens[1] is screen2
