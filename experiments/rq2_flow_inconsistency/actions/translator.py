import xml.etree.ElementTree as ET
from typing import Literal

import uiautomator2 as u2
from uiautomator2 import Device, UiObject

from guipilot.entities import Bbox, Screen

from .automator import Automator


class Translator(Automator):
    def __init__(
        self,
        screen: Screen,
        device: Device = u2.connect("192.168.240.112:5555"),
    ) -> None:
        """Translator extends the Automator and
        translates actions provided by the VLM agent into executable actions
        """
        super().__init__(device)
        self.widgets = screen.widgets

    def _id_to_locator(self, id: int) -> tuple[dict, Bbox]:
        widget = self.widgets[id]
        cx = (widget.bbox.xmin + widget.bbox.xmax) // 2
        cy = (widget.bbox.ymin + widget.bbox.ymax) // 2
        locator = {"x": cx, "y": cy}
        return locator, widget.bbox

    def back(self):
        super().back()
        return []

    def click(self, id: int) -> list[Bbox]:
        locator, bbox = self._id_to_locator(id)
        super().click(locator)
        return [bbox]

    def long_click(self, id: int) -> list[Bbox]:
        locator, bbox = self._id_to_locator(id)
        super().long_click(locator)
        return [bbox]

    def send_keys(self, text: str):
        super().send_keys(text)
        return []

    def scroll(self, direction: Literal["left", "right", "up", "down"]):
        super().scroll(direction)
        return []

    def swipe(
        self, id: int, direction: Literal["left", "right", "up", "down"]
    ) -> list[Bbox]:
        """Swipe from the center of the widget to its edge."""

        # Function to parse bounds from the hierarchy dump
        def _parse_bounds(bounds_str: str):
            bounds_str = bounds_str.strip("[]")
            left_top, right_bottom = bounds_str.split("][")
            left_top = list(map(int, left_top.split(",")))
            right_bottom = list(map(int, right_bottom.split(",")))
            return left_top, right_bottom

        # Function to find the element by the given point (x, y)
        def _find_element_by_point(hierarchy_xml: str, x: int, y: int):
            root = ET.fromstring(hierarchy_xml)
            for node in root.iter("node"):
                bounds = node.attrib.get("bounds")
                if bounds:
                    (left_top, right_bottom) = _parse_bounds(bounds)
                    if (
                        left_top[0] <= x <= right_bottom[0]
                        and left_top[1] <= y <= right_bottom[1]
                    ):
                        return node
            return None

        locator, bbox = self._id_to_locator(id)
        xml = self.device.dump_hierarchy()
        element = _find_element_by_point(xml, locator["x"], locator["y"])
        if isinstance(element, UiObject):
            element.swipe(direction)
        return [bbox]

    def drag(self, id1: int, id2: int) -> list[Bbox]:
        """Drag widget to the center of another widget."""
        locator1, bbox1 = self._id_to_locator(id1)
        locator2, bbox2 = self._id_to_locator(id2)
        super().drag(locator1, locator2)
        return [bbox1, bbox2]
