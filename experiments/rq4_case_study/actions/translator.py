from typing import Literal

from guipilot.entities import Bbox, Screen


class Translator:
    def __init__(self, screen: Screen) -> None:
        self.widgets = screen.widgets

    def click(self, id: int) -> list[Bbox]:
        return [self.widgets[id].bbox]

    def long_click(self, id: int) -> list[Bbox]:
        return [self.widgets[id].bbox]

    def send_keys(self, text: str):
        return []

    def scroll(self, direction: Literal["left", "right", "up", "down"]):
        return []

    def swipe(
        self, id: int, direction: Literal["left", "right", "up", "down"]
    ) -> list[Bbox]:
        return [self.widgets[id].bbox]

    def drag(self, id1: int, id2: int) -> list[Bbox]:
        return [self.widgets[id1].bbox, self.widgets[id2].bbox]
