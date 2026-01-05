from collections import namedtuple
from enum import Enum

# Define a named tuple for bounding box coordinates
Bbox = namedtuple("Bbox", ["xmin", "ymin", "xmax", "ymax"])
"""
Named tuple representing the rectangular coordinates of a GUI element.

Attributes:
    xmin (int/float): The minimum x-coordinate (left edge).
    ymin (int/float): The minimum y-coordinate (top edge).
    xmax (int/float): The maximum x-coordinate (right edge).
    ymax (int/float): The maximum y-coordinate (bottom edge).
"""


class Inconsistency(Enum):
    """
    Enumeration of the types of discrepancies detected between GUI screens.

    This enum categorizes the differences found during the comparison of a mockup
    design and its actual implementation.
    """

    BBOX = 0
    """Indicates a spatial or layout discrepancy where the position or size 
    of a widget does not match the expected design."""

    TEXT = 1
    """Indicates a content discrepancy where the textual information within 
    a widget differs from the expected value."""

    COLOR = 2
    """Indicates a visual discrepancy where the color distribution or 
    perceptual appearance of a widget does not match the design."""
