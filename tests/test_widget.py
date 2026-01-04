from guipilot.entities.constants import Bbox
from guipilot.entities.widget import Widget, WidgetType


def test_widget_properties():
    bbox = Bbox(10, 20, 110, 120)
    widget = Widget(type=WidgetType.TEXT_BUTTON, bbox=bbox)

    assert widget.width == 100
    assert widget.height == 100
    assert widget.area == 10000
    assert widget.center == (60, 70)


def test_widget_with_text():
    bbox = Bbox(0, 0, 50, 50)
    widget = Widget(
        type=WidgetType.TEXT_VIEW,
        bbox=bbox,
        texts=["Login"],
        text_bboxes=[Bbox(5, 5, 45, 45)],
    )

    assert widget.texts == ["Login"]
    assert len(widget.text_bboxes) == 1
    assert widget.text_bboxes[0].xmin == 5
