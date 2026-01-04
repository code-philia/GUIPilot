from collections import namedtuple
from enum import Enum

Bbox = namedtuple("Bbox", ["xmin", "ymin", "xmax", "ymax"])


class Inconsistency(Enum):
    BBOX = 0
    TEXT = 1
    COLOR = 2
