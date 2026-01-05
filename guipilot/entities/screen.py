from __future__ import annotations

import typing
from dataclasses import dataclass, field

import cv2
import numpy as np

from guipilot.models import OCR, Detector

from .constants import Bbox
from .widget import Widget, WidgetType

if typing.TYPE_CHECKING:
    from guipilot.checker import ScreenChecker
    from guipilot.matcher import WidgetMatcher

    from .screen import Screen

# Initialize external services for OCR and widget detection
ocr = OCR(service_url="http://localhost:5000/detect")
detector = Detector(service_url="http://localhost:6000/detect")


@dataclass
class Screen:
    """
    Data container representing a mobile application screen.

    This class encapsulates the screen's visual data (image) and its constituent
    UI components (widgets). It provides high-level methods for automated
    widget detection, text extraction, and consistency validation.

    Attributes:
        image (np.ndarray): The RGB or BGR screenshot of the screen.
        widgets (dict[int, Widget]): A mapping of unique integer IDs to
            their corresponding Widget entities.
    """

    image: np.ndarray
    widgets: dict[int, Widget] = field(default_factory=dict)

    def detect(self) -> None:
        """
        Extracts UI components from the screen using an object detection model.

        This method sends the screen image to a remote detector service,
        identifies bounding boxes and widget types, and populates the
        `self.widgets` attribute with the results.

        Note:
            The internal helper `_to_bbox` ensures detected points are
            correctly cast to integer-based Bbox instances.
        """

        def _to_bbox(points: np.ndarray) -> Bbox:
            """Converts raw detection points to a Bbox object."""
            xmin, ymin, xmax, ymax = points
            return Bbox(int(xmin), int(ymin), int(xmax), int(ymax))

        bboxes, widget_types = detector(self.image)
        self.widgets = {
            i: Widget(type=WidgetType(widget_type), bbox=_to_bbox(bbox))
            for i, (bbox, widget_type) in enumerate(zip(bboxes, widget_types))
        }

        assert len(self.widgets) == len(bboxes) == len(widget_types)

    def ocr(self) -> None:
        """
        Extracts text content from each identified widget using OCR.

        Iterates through all widgets in the screen, crops their respective
        image regions, and sends them to the OCR service to identify
        textual labels and their internal bounding boxes.

        Note:
            Input images are converted from BGR to RGB format before
            processing to meet OCR service requirements.
        """
        image = np.array(self.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        widgets = self.widgets.values()
        for widget in widgets:
            xmin, ymin, xmax, ymax = widget.bbox
            widget_image = image[ymin:ymax, xmin:xmax].copy()
            try:
                widget.texts, widget.text_bboxes = ocr(widget_image)
            except Exception as e:
                # Logs errors related to OCR service communication or image cropping
                print(e)
                print(self.image.shape)
                print(widget.bbox)

    def check(
        self, target: Screen, matcher: WidgetMatcher, checker: ScreenChecker
    ) -> tuple[set, float]:
        """
        Identifies inconsistencies between this screen and a target screen.

        This is a pipeline method that orchestrates widget matching across
        screens followed by a detailed consistency check for each pair.

        Args:
            target (Screen): The screen entity to compare against (e.g., the
                actual implementation vs. this mockup).
            matcher (WidgetMatcher): The algorithm instance used to pair
                corresponding widgets across both screens.
            checker (ScreenChecker): The algorithm instance used to evaluate
                consistency (BBox, Text, Color) for matched pairs.

        Returns:
            tuple[set, float]: A tuple containing:
                - set: A set of identified inconsistencies (from Inconsistency enum).
                - float: The total time taken to complete the check in seconds.
        """
        pairs = matcher.match(self, target)
        inconsistencies, time_taken = checker.check(self, target, pairs)
        return inconsistencies, time_taken
