from __future__ import annotations

import typing
from dataclasses import dataclass, field
from enum import Enum

if typing.TYPE_CHECKING:
    from .constants import Bbox


class WidgetType(Enum):
    TEXT_BUTTON = "textbutton"
    ICON_BUTTON = "iconbutton"
    COMBINED_BUTTON = "combinedbutton"
    INPUT_BOX = "inputbox"
    TEXT_VIEW = "textview"
    IMAGE_VIEW = "imageview"
    CHART = "chart"
    SLIDER = "slider"


@dataclass
class Widget:
    type: WidgetType
    bbox: Bbox
    texts: list[str] = field(default_factory=list)
    text_bboxes: list[Bbox] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.bbox.xmax - self.bbox.xmin

    @property
    def height(self) -> float:
        return self.bbox.ymax - self.bbox.ymin

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        cx = (self.bbox.xmin + self.bbox.xmax) / 2
        cy = (self.bbox.ymin + self.bbox.ymax) / 2
        return cx, cy
