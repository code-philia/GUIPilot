import copy
import os

import cv2
import numpy as np
import supervision as sv
from supervision import Detections

from guipilot.entities import Screen


def _get_one_image(img_list: list[np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple images into a single horizontal image.
    Pads images with zeros to match the maximum height among all images.

    Args:
        img_list: List of images to concatenate (np.ndarray)

    Returns:
        Concatenated image (np.ndarray)
    """
    max_height = 0
    total_width = 0
    for img in img_list:
        if img.shape[0] > max_height:
            max_height = img.shape[0]
        total_width += img.shape[1]

    # Create empty canvas with sufficient size
    final_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    current_x = 0

    for image in img_list:
        # Pad image height to match max_height
        if image.shape[0] < max_height:
            pad_height = max_height - image.shape[0]
            image = np.vstack((image, np.zeros((pad_height, image.shape[1], 3))))
        final_image[:, current_x : current_x + image.shape[1], :] = image
        current_x += image.shape[1]
    return final_image


def visualize_inconsistencies(
    s1: Screen,
    s2: Screen,
    pairs: list[tuple],
    inconsistencies: list[tuple],
    save_path: str = None,  # Optional: save path for the image
) -> np.ndarray:
    """
    Visualize widget matching pairs and inconsistencies between two screens.

    Args:
        s1, s2: Two screens to compare
        pairs: List of matched widget ID pairs (e.g., [(id1, id2), ...])
        inconsistencies: List of inconsistent widget pairs with types
        save_path: If provided, save the image to this path (e.g., "./vis/1.jpg")

    Returns:
        Concatenated visualization image (np.ndarray)
    """
    # Initialize annotators with specific styles (consistent with original code)
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

    # Classify bounding boxes into categories
    s1_bboxes = {"paired": {}, "paired_inconsistent": {}, "unpaired": {}}
    s2_bboxes = {"paired": {}, "paired_inconsistent": {}, "unpaired": {}}
    paired_inconsistent = set()

    # Process inconsistencies first
    for inconsistency in inconsistencies:
        id1, id2 = inconsistency[:2]
        # Extract bbox coordinates if IDs exist
        if id1 is not None:
            xmin1, ymin1, xmax1, ymax1 = s1.widgets[id1].bbox
        if id2 is not None:
            xmin2, ymin2, xmax2, ymax2 = s2.widgets[id2].bbox

        # Categorize based on presence of IDs
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

    # Process normal pairs (excluding inconsistent ones)
    for pair in pairs:
        if pair in paired_inconsistent:
            continue
        id1, id2 = pair
        xmin1, ymin1, xmax1, ymax1 = s1.widgets[id1].bbox
        xmin2, ymin2, xmax2, ymax2 = s2.widgets[id2].bbox
        s1_bboxes["paired"][id1] = [int(xmin1), int(ymin1), int(xmax1), int(ymax1)]
        s2_bboxes["paired"][id2] = [int(xmin2), int(ymin2), int(xmax2), int(ymax2)]

    # Annotate first screen (use deepcopy for rq1 compatibility)
    s1_image = copy.deepcopy(s1.image)
    for (_, bboxes), annotator in zip(s1_bboxes.items(), annotators):
        if len(bboxes) == 0:
            continue
        detections = Detections(np.array([bbox for bbox in bboxes.values()]))
        annotator.annotate(s1_image, detections)
        label_annotator.annotate(
            s1_image, detections, labels=[f"{i}" for i in bboxes.keys()]
        )

    # Annotate second screen (use deepcopy for rq1 compatibility)
    s2_image = copy.deepcopy(s2.image)
    for (_, bboxes), annotator in zip(s2_bboxes.items(), annotators):
        if len(bboxes) == 0:
            continue
        detections = Detections(np.array([bbox for bbox in bboxes.values()]))
        annotator.annotate(s2_image, detections)
        label_annotator.annotate(
            s2_image, detections, labels=[f"{i}" for i in bboxes.keys()]
        )

    # Concatenate images
    final_image = _get_one_image([s1_image, s2_image])

    # Save image if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, final_image)

    return final_image
