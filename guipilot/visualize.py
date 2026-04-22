import copy

import cv2
import numpy as np
import supervision as sv
from supervision import Detections

from guipilot.entities import Screen


# Colours match the RQ1 convention
_GREEN  = sv.Color.GREEN   # matched, consistent
_YELLOW = sv.Color.YELLOW  # matched, inconsistent
_RED    = sv.Color.RED     # unmatched (missing / excess)

_LOOKUP = sv.ColorLookup.INDEX


def _make_annotators(color: sv.Color):
    box = sv.BoxAnnotator(color=color, thickness=2, color_lookup=_LOOKUP)
    label = sv.LabelAnnotator(
        color=color,
        text_color=sv.Color.WHITE,
        color_lookup=_LOOKUP,
        text_position=sv.Position.TOP_LEFT,
        text_padding=2,
    )
    return box, label


def _annotate(image: np.ndarray, bboxes: dict, color: sv.Color) -> np.ndarray:
    if not bboxes:
        return image
    xyxy = np.array(list(bboxes.values()), dtype=float)
    labels = [str(k) for k in bboxes]
    detections = Detections(xyxy=xyxy)
    box_ann, label_ann = _make_annotators(color)
    image = box_ann.annotate(image, detections)
    image = label_ann.annotate(image, detections, labels=labels)
    return image


def _hstack(imgs: list[np.ndarray]) -> np.ndarray:
    max_h = max(img.shape[0] for img in imgs)
    padded = []
    for img in imgs:
        pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        padded.append(np.vstack([img, pad]))
    return np.hstack(padded)


def visualize_inconsistencies(
    mock_screen: Screen,
    real_screen: Screen,
    pairs: list[tuple],
    inconsistencies,
    out_path: str,
) -> None:
    """Save a side-by-side annotated image (mock | real).

    Color legend:
      Green  — matched widget pairs with no inconsistency
      Yellow — matched pairs with a detected inconsistency
      Red    — unmatched widgets (missing in real / excess in real)
    """
    s1_paired:   dict[int, tuple] = {}
    s1_mismatch: dict[int, tuple] = {}
    s1_unpaired: dict[int, tuple] = {}
    s2_paired:   dict[int, tuple] = {}
    s2_mismatch: dict[int, tuple] = {}
    s2_unpaired: dict[int, tuple] = {}

    paired_inconsistent: set[tuple] = set()
    for item in inconsistencies:
        id1, id2 = item[0], item[1]
        if id1 is not None and id2 is not None:
            if id1 not in mock_screen.widgets or id2 not in real_screen.widgets:
                continue
            b1 = mock_screen.widgets[id1].bbox
            b2 = real_screen.widgets[id2].bbox
            s1_mismatch[id1] = (int(b1[0]), int(b1[1]), int(b1[2]), int(b1[3]))
            s2_mismatch[id2] = (int(b2[0]), int(b2[1]), int(b2[2]), int(b2[3]))
            paired_inconsistent.add((id1, id2))
        elif id1 is not None:
            if id1 not in mock_screen.widgets:
                continue
            b1 = mock_screen.widgets[id1].bbox
            s1_unpaired[id1] = (int(b1[0]), int(b1[1]), int(b1[2]), int(b1[3]))
        else:
            if id2 not in real_screen.widgets:
                continue
            b2 = real_screen.widgets[id2].bbox
            s2_unpaired[id2] = (int(b2[0]), int(b2[1]), int(b2[2]), int(b2[3]))

    for id1, id2 in pairs:
        if (id1, id2) in paired_inconsistent:
            continue
        b1 = mock_screen.widgets[id1].bbox
        b2 = real_screen.widgets[id2].bbox
        s1_paired[id1] = (int(b1[0]), int(b1[1]), int(b1[2]), int(b1[3]))
        s2_paired[id2] = (int(b2[0]), int(b2[1]), int(b2[2]), int(b2[3]))

    s1_img = copy.deepcopy(mock_screen.image)
    s2_img = copy.deepcopy(real_screen.image)

    for bboxes, color, img_ref in [
        (s1_paired,   _GREEN,  's1'), (s1_mismatch, _YELLOW, 's1'), (s1_unpaired, _RED, 's1'),
        (s2_paired,   _GREEN,  's2'), (s2_mismatch, _YELLOW, 's2'), (s2_unpaired, _RED, 's2'),
    ]:
        target = s1_img if img_ref == 's1' else s2_img
        result = _annotate(target, bboxes, color)
        if img_ref == 's1':
            s1_img = result
        else:
            s2_img = result

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, _hstack([s1_img, s2_img]))
