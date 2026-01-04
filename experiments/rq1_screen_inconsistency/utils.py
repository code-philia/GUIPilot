import copy
import json
import os

import cv2
import numpy as np
import supervision as sv
from supervision import Detections

from guipilot.entities import Bbox, Inconsistency, Screen, Widget, WidgetType


def visualize_inconsistencies(
    s1: Screen,
    s2: Screen,
    pairs: list[tuple],
    inconsistencies: list[tuple],
    path: str,
    filename: str,
):
    def _get_one_image(img_list: list[np.ndarray]):
        max_height = 0
        total_width = 0  # padding
        for img in img_list:
            if img.shape[0] > max_height:
                max_height = img.shape[0]
            total_width += img.shape[1]

        # create a new array with a size large enough to contain all the images
        final_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)

        current_x = 0  # keep track of where your current image was last placed in the y coordinate
        for image in img_list:
            # add an image to the final array and increment the y coordinate
            image = np.vstack(
                (image, np.zeros((max_height - image.shape[0], image.shape[1], 3)))
            )
            final_image[:, current_x : current_x + image.shape[1], :] = image
            current_x += image.shape[1]
        return final_image

    annotators = [
        sv.BoxAnnotator(
            color=sv.Color.GREEN, thickness=2, color_lookup=sv.ColorLookup.INDEX
        ),
        sv.BoxAnnotator(
            color=sv.Color.YELLOW, thickness=2, color_lookup=sv.ColorLookup.INDEX
        ),
        sv.BoxAnnotator(
            color=sv.Color.RED, thickness=2, color_lookup=sv.ColorLookup.INDEX
        ),
    ]
    label_annotator = sv.LabelAnnotator(
        color=sv.Color.BLACK,
        text_color=sv.Color.WHITE,
        color_lookup=sv.ColorLookup.INDEX,
        text_position=sv.Position.TOP_LEFT,
        text_padding=1,
    )

    s1_bboxes = {"paired": {}, "paired_inconsistent": {}, "unpaired": {}}
    s2_bboxes = {"paired": {}, "paired_inconsistent": {}, "unpaired": {}}

    paired_inconsistent = set()
    for inconsistency in inconsistencies:
        id1, id2 = inconsistency[:2]
        if id1 is not None:
            xmin1, ymin1, xmax1, ymax1 = s1.widgets[id1].bbox
        if id2 is not None:
            xmin2, ymin2, xmax2, ymax2 = s2.widgets[id2].bbox
        if id1 is not None and id2 is not None:
            s1_bboxes["paired_inconsistent"][id1] = [
                int(xmin1),
                int(ymin1),
                int(xmax1),
                int(ymax1),
            ]
            s2_bboxes["paired_inconsistent"][id2] = [
                int(xmin2),
                int(ymin2),
                int(xmax2),
                int(ymax2),
            ]
            paired_inconsistent.add((id1, id2))
        elif id1 is not None:
            s1_bboxes["unpaired"][id1] = [
                int(xmin1),
                int(ymin1),
                int(xmax1),
                int(ymax1),
            ]
        elif id2 is not None:
            s2_bboxes["unpaired"][id2] = [
                int(xmin2),
                int(ymin2),
                int(xmax2),
                int(ymax2),
            ]

    for pair in pairs:
        if pair in paired_inconsistent:
            continue
        id1, id2 = pair
        xmin1, ymin1, xmax1, ymax1 = s1.widgets[id1].bbox
        xmin2, ymin2, xmax2, ymax2 = s2.widgets[id2].bbox
        s1_bboxes["paired"][id1] = [int(xmin1), int(ymin1), int(xmax1), int(ymax1)]
        s2_bboxes["paired"][id2] = [int(xmin2), int(ymin2), int(xmax2), int(ymax2)]

    s1_image = copy.deepcopy(s1.image)
    for (name, bboxes), annotator in zip(s1_bboxes.items(), annotators):
        if len(bboxes) == 0:
            continue
        detections = Detections(np.array([bbox for bbox in bboxes.values()]))
        annotator.annotate(s1_image, detections)
        label_annotator.annotate(
            s1_image, detections, labels=[f"{i}" for i in bboxes.keys()]
        )

    s2_image = copy.deepcopy(s2.image)
    for (name, bboxes), annotator in zip(s2_bboxes.items(), annotators):
        if len(bboxes) == 0:
            continue
        detections = Detections(np.array([bbox for bbox in bboxes.values()]))
        annotator.annotate(s2_image, detections)
        label_annotator.annotate(
            s2_image, detections, labels=[f"{i}" for i in bboxes.keys()]
        )

    os.makedirs(f"./visualize/{path}", exist_ok=True)
    image = _get_one_image([s1_image, s2_image])
    cv2.imwrite(f"./visualize/{path}/{filename}.jpg", image)


def remove_overlapping_widgets(widgets: dict[int, Widget]) -> dict[int, Widget]:
    """Pre-processing: Filter out overlapping widgets"""

    def is_contained(bbox1: Bbox, bbox2: Bbox):
        """Check if bbox1 is completely inside bbox2."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        return (
            x1_min >= x2_min
            and y1_min >= y2_min
            and x1_max <= x2_max
            and y1_max <= y2_max
        )

    def is_high_iou(bbox1: Bbox, bbox2: Bbox):
        """Check if bboxes have IOU above threshold."""
        xa, ya = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
        xb, yb = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])
        intersection = abs(max((xb - xa, 0)) * max((yb - ya), 0))
        boxa = abs((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
        boxb = abs((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
        iou = intersection / (boxa + boxb - intersection)
        return iou >= 0.95

    refined = []

    for i, widget_i in widgets.items():
        contained = False
        for j, widget_j in widgets.items():
            if i == j:
                continue
            if is_contained(widget_i.bbox, widget_j.bbox) or is_high_iou(
                widget_i.bbox, widget_j.bbox
            ):
                contained = True
                break

        if not contained:
            refined.append(widget_i)

    return {i: widget for i, widget in enumerate(refined)}


def load_screen(image_path: str) -> Screen:
    def _points_to_bbox(points: list[list[int, int]]) -> Bbox:
        """Pre-processing: Convert unordered points to bounding boxes"""
        p1 = points[0]
        p2 = points[1]
        xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0])
        ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
        assert xmin < xmax and ymin < ymax
        return Bbox(int(xmin), int(ymin), int(xmax), int(ymax))

    label_path = image_path.replace(".jpg", ".json")
    label = json.load(open(label_path, encoding="utf-8"))
    image = cv2.imread(image_path)
    bboxes = [_points_to_bbox(shape["points"]) for shape in label["shapes"]]
    labels = [WidgetType(shape["label"]) for shape in label["shapes"]]
    sorted_indices = np.lexsort(
        ([bbox.xmin for bbox in bboxes], [bbox.ymin for bbox in bboxes])
    )
    sorted_bboxes = [bboxes[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    # Prepare screen its and widgets
    widgets = {
        i: Widget(type=widget_type, bbox=bbox)
        for i, (widget_type, bbox) in enumerate(zip(sorted_labels, sorted_bboxes))
    }
    widgets = remove_overlapping_widgets(widgets)
    return Screen(image, widgets)


def filter_swapped_predictions(y_pred: set, y_true: set, s1: Screen, s2: Screen) -> set:
    def _bbox_overlap(bbox1, bbox2):
        return not (
            bbox1[2] < bbox2[0]
            or bbox2[2] < bbox1[0]
            or bbox1[3] < bbox2[1]
            or bbox2[3] < bbox1[1]
        )

    y_pred = copy.deepcopy(y_pred)

    for item in y_true:
        id1, id2, _ = item

        alternative1 = [(id1, None), (None, id1), (id2, id2, Inconsistency.BBOX)]
        alternative2 = [(id2, None), (None, id2), (id1, id1, Inconsistency.BBOX)]

        # (x, None), (y, y, bbox), (None, x) -> (x, y, bbox), (y, x, bbox)
        if all([x in y_pred for x in alternative1]):
            for x in alternative1:
                y_pred.discard(x)
            y_pred.add((id1, id2, Inconsistency.BBOX))
            y_pred.add((id2, id1, Inconsistency.BBOX))

        # (y, None), (x, x, bbox), (None, y) -> (x, y, bbox), (y, x, bbox)
        if all([x in y_pred for x in alternative2]):
            for x in alternative2:
                y_pred.discard(x)
            y_pred.add((id1, id2, Inconsistency.BBOX))
            y_pred.add((id2, id1, Inconsistency.BBOX))

        if (id1, id2, Inconsistency.TEXT) in y_pred:
            y_pred.discard((id1, id2, Inconsistency.TEXT))
            y_pred.add((id1, id2, Inconsistency.BBOX))

        if (id1, id2, Inconsistency.COLOR) in y_pred:
            y_pred.discard((id1, id2, Inconsistency.COLOR))
            y_pred.add((id1, id2, Inconsistency.BBOX))

    # ignore inconsistencies from overlap bboxes
    overlap_ids = set()
    swapped_ids = [item[1] for item in y_true]
    for item in y_pred:
        id1 = item[1]
        if id1 is None or id1 in swapped_ids:
            continue
        for swapped_id in swapped_ids:
            if _bbox_overlap(s2.widgets[id1].bbox, s2.widgets[swapped_id].bbox):
                overlap_ids.add(id1)

    swapped_ids = [item[0] for item in y_true]
    for item in y_pred:
        id1 = item[0]
        if id1 is None or id1 in swapped_ids:
            continue
        for swapped_id in swapped_ids:
            if _bbox_overlap(s1.widgets[id1].bbox, s1.widgets[swapped_id].bbox):
                overlap_ids.add(id1)

    for overlap_id in overlap_ids:
        y_pred.discard((overlap_id, overlap_id, Inconsistency.TEXT))
        y_pred.discard((overlap_id, overlap_id, Inconsistency.COLOR))
        y_pred.discard((overlap_id, overlap_id, Inconsistency.BBOX))

    return y_pred


def filter_overlap_predictions(y_pred: set, y_true: set, s1: Screen, s2: Screen) -> set:
    """Ignore those child/parent widgets affected (overlap) by the ground truth mutated widget"""

    def _bbox_overlap(bbox1, bbox2):
        return not (
            bbox1[2] < bbox2[0]
            or bbox2[2] < bbox1[0]
            or bbox1[3] < bbox2[1]
            or bbox2[3] < bbox1[1]
        )

    overlap_ids = set()
    if s1 is not None:
        mutated_ids = [item[0] for item in y_true]
        s = s1

    elif s2 is not None:
        mutated_ids = [item[1] for item in y_true]
        s = s2

    for item in y_pred:
        id1, id2 = item[0], item[1]
        if id1 != id2:
            continue
        if id1 in mutated_ids:
            continue
        for id in mutated_ids:
            if _bbox_overlap(s.widgets[id].bbox, s.widgets[id1].bbox):
                overlap_ids.add(id1)

    for overlap_id in overlap_ids:
        y_pred.discard((overlap_id, overlap_id, Inconsistency.TEXT))
        y_pred.discard((overlap_id, overlap_id, Inconsistency.COLOR))
        y_pred.discard((overlap_id, overlap_id, Inconsistency.BBOX))

    return y_pred


def filter_text(y_pred: set, y_true: set, s1: Screen, s2: Screen) -> set:
    """For change widgets color, ignore text inconsistencies"""
    y_pred = filter_overlap_predictions(y_pred, y_true, s1, s2)
    target_ids = set([item[0] for item in y_true])

    for id in target_ids:
        y_pred.discard((id, id, Inconsistency.TEXT))

    return y_pred


def filter_color(y_pred: set, y_true: set, s1: Screen, s2: Screen) -> set:
    """For change widgets text, ignore color inconsistencies"""
    y_pred = filter_overlap_predictions(y_pred, y_true, s1, s2)
    target_ids = set([item[0] for item in y_true])

    for id in target_ids:
        y_pred.discard((id, id, Inconsistency.COLOR))

    return y_pred


def convert_inconsistencies(inconsistencies: set) -> tuple[set, set]:
    """Convert standard inconsistency representation to editdistance representation

    1. Deletion: (a, None) -> (delete, a)

    2. Insertion: (None, b) -> (insert, b)

    3. Substitution:
        3.1 Text (a, b, TEXT) -> (substitute.TEXT, a, b)
        3.2 Color (a, b, COLOR) -> (substitute.COLOR, a, b)

    4. Swapping: can either be
        4.1 2 substitutions: (substitute, a1, b1), (substitute, a2, b2)
        4.2 2 insertion-deletions: (delete, a1), (insert, b1), (delete, a2), (insert, b2)
        4.3 or 1 substitution + 1 insertion-deletion

        filter_swapped_predictions() standardizes all to 4.1
        (a1, b1, BBOX), (a1, b2, BBOX) -> (substitute.SWAP, a1, b1), (substitute.SWAP, a2, b2)
    """
    result = set()
    for inconsistency in inconsistencies:
        id1, id2 = inconsistency[0], inconsistency[1]
        i_type = inconsistency[2] if len(inconsistency) > 2 else None

        if id1 is None:
            result.add(("insert", id2))

        elif id2 is None:
            result.add(("delete", id1))

        elif i_type == Inconsistency.COLOR:
            result.add(("substitute.color", id1, id2))

        elif i_type == Inconsistency.TEXT:
            result.add(("substitute.text", id1, id2))

        elif i_type == Inconsistency.BBOX and (id2, id1, i_type) in inconsistencies:
            result.add(("substitute.swap", id1, id2))

        elif i_type == Inconsistency.BBOX:
            result.add(("substitute.bbox", id1, id2))

    return result
