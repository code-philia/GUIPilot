import os
import re
import json
import copy
from functools import partial
from typing import Iterable
from timeit import default_timer as timer

import cv2
import jsbeautifier
import numpy as np
import supervision as sv
from PIL import Image
from supervision import Detections

from actions import Translator
from guipilot.agent import Agent
from guipilot.entities import Screen
from guipilot.matcher import WidgetMatcher
from guipilot.entities import Screen


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def get_screen(process_path: str, filename: str, tolerate_failure: bool = False) -> Screen | None:
    image_path = os.path.join(process_path, filename)
    image: np.ndarray = cv2.imread(image_path)
    image = image_resize(image, width=1080)
    screen = Screen(image)
    try:
        screen.detect()
        screen.ocr()
    except Exception as exc:
        if tolerate_failure:
            print(f"[warning] Failed to run detector/OCR for {image_path}: {exc}")
            return None
        raise
    return screen


def get_scores(mock: Screen, real: Screen, matcher: WidgetMatcher) -> tuple[list, float, float]:
    """Get consistency scores for the current (real) screen as compared to mock screen.

    Args:
        mock, real: The screens to check against.
        matcher: An algorithm that pairs matching widgets on both screens.
        checker: An algorithm that checks the consistency of paired widgets.

    Returns:
        list[tuple]: A list of matched pairs.
        float: A tuple of different calculated scores.
        float: A tuple of time taken for calculating scores.
    """
    start_time = timer()
    pairs, scores, _ = matcher.match(mock, real)
    score = sum(scores) / len(real.widgets)
    time = 1000 * (timer() - start_time)

    return pairs, score, time


def get_action_completion(agent: Agent, screen: Screen, mock_actions: list[str]):
    image = annotate_screen(screen)
    base_path = os.path.abspath(os.path.dirname(__file__))
    prompt_path = os.path.join(base_path, "action_completion.user.prompt")
    prompt = open(prompt_path).read()
    prompt = prompt.replace("{ACTION_DESCRIPTION}", ", ".join(mock_actions))
    response = agent(prompt, [image])

    actions, action_names = [], []
    translator = Translator(screen)
    matches = re.findall(r"(\w+)\((.*)\)", response)
    for method_name, params in matches:
        method = getattr(translator, method_name, None)
        param_list = eval(f"({params})")
        if not isinstance(param_list, tuple):
            param_list = (param_list,)
            
        if method is not None: 
            action = partial(method, *param_list)
            actions.append(action)
            action_names.append(method_name)

    actions_raw = [f"{method_name}({[params]})" for method_name, params in matches]
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image, action_names, actions_raw, actions


def get_report(process_id, screen_id, match_time, check_time, pairs, inconsistencies, action_correct, action_trials) -> str:
    mapping: dict[tuple, list] = {}
    for inc in inconsistencies:
        if len(inc) == 3:
            mapping.setdefault((inc[0], inc[1]), []).append(str(inc[2]))

    report = {
        "process": process_id,
        "screen": screen_id,
        "match_time": match_time,
        "check_time": check_time,
        "pairs": [
            {
                "ids": [id1, id2],
                **({"inconsistencies": mapping[(id1, id2)]} if (id1, id2) in mapping else {})
            }
            for (id1, id2) in pairs
        ],
        "missing": sorted([inc[0] for inc in inconsistencies if inc[1] is None]),
        "extra": sorted([inc[1] for inc in inconsistencies if inc[0] is None]),
        "action_correct": action_correct,
        "action_trials": action_trials
    }

    options = jsbeautifier.default_options()
    options.indent_size = 2
    return jsbeautifier.beautify(json.dumps(report, ensure_ascii=False), options)


def visualize(s1: Screen, s2: Screen, pairs: list[tuple], inconsistencies: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            image = np.vstack((image, np.zeros((max_height - image.shape[0], image.shape[1], 3))))
            final_image[:, current_x:current_x + image.shape[1], :] = image
            current_x += image.shape[1]
        return final_image

    annotators = [
        sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2, color_lookup=sv.ColorLookup.INDEX),
        sv.BoxAnnotator(color=sv.Color.YELLOW, thickness=2, color_lookup=sv.ColorLookup.INDEX),
        sv.BoxAnnotator(color=sv.Color.RED, thickness=2, color_lookup=sv.ColorLookup.INDEX)
    ]
    label_annotator = sv.LabelAnnotator(color=sv.Color.BLACK, text_color=sv.Color.WHITE, color_lookup=sv.ColorLookup.INDEX, text_position=sv.Position.TOP_LEFT, text_padding=1)

    s1_bboxes = {"paired": {}, "paired_inconsistent": {}, "unpaired": {}}
    s2_bboxes = {"paired": {}, "paired_inconsistent": {}, "unpaired": {}}

    paired_inconsistent = set()
    for inconsistency in inconsistencies:
        id1, id2 = inconsistency[:2]
        if id1 is not None: xmin1, ymin1, xmax1, ymax1 = s1.widgets[id1].bbox
        if id2 is not None: xmin2, ymin2, xmax2, ymax2 = s2.widgets[id2].bbox
        if id1 is not None and id2 is not None:
            s1_bboxes["paired_inconsistent"][id1] = [int(xmin1), int(ymin1), int(xmax1), int(ymax1)]
            s2_bboxes["paired_inconsistent"][id2] = [int(xmin2), int(ymin2), int(xmax2), int(ymax2)]
            paired_inconsistent.add((id1, id2))
        elif id1 is not None:
            s1_bboxes["unpaired"][id1] = [int(xmin1), int(ymin1), int(xmax1), int(ymax1)]
        elif id2 is not None:
            s2_bboxes["unpaired"][id2] = [int(xmin2), int(ymin2), int(xmax2), int(ymax2)]

    for pair in pairs:
        if pair in paired_inconsistent: continue
        id1, id2 = pair
        xmin1, ymin1, xmax1, ymax1 = s1.widgets[id1].bbox
        xmin2, ymin2, xmax2, ymax2 = s2.widgets[id2].bbox
        s1_bboxes["paired"][id1] = [int(xmin1), int(ymin1), int(xmax1), int(ymax1)]
        s2_bboxes["paired"][id2] = [int(xmin2), int(ymin2), int(xmax2), int(ymax2)]

    s1_image = copy.deepcopy(s1.image)
    for (_, bboxes), annotator in zip(s1_bboxes.items(), annotators):
        if len(bboxes) == 0: continue
        detections = Detections(np.array([bbox for bbox in bboxes.values()]))
        annotator.annotate(s1_image, detections)
        label_annotator.annotate(s1_image, detections, labels=[f"{i}" for i in bboxes.keys()])

    s2_image = copy.deepcopy(s2.image)
    for (_, bboxes), annotator in zip(s2_bboxes.items(), annotators):
        if len(bboxes) == 0: continue
        detections = Detections(np.array([bbox for bbox in bboxes.values()]))
        annotator.annotate(s2_image, detections)
        label_annotator.annotate(s2_image, detections, labels=[f"{i}" for i in bboxes.keys()])

    image = _get_one_image([s1.image, s2.image])
    bbox_image = _get_one_image([s1_image, s2_image])
    
    match_image = copy.deepcopy(bbox_image)
    _, x_shift, _ = s1_image.shape
    for pair in pairs:
        id1, id2 = pair
        if id1 is not None and id2 is not None:
            xmin1, ymin1, xmax1, ymax1 = s1.widgets[id1].bbox
            xmin2, ymin2, xmax2, ymax2 = s2.widgets[id2].bbox
            xc1 = (xmin1 + xmax1) // 2
            yc1 = (ymin1 + ymax1) // 2
            xc2 = (xmin2 + xmax2) // 2
            yc2 = (ymin2 + ymax2) // 2
            cv2.line(match_image, (xc1, yc1), (x_shift + xc2, yc2), (255, 0, 0), thickness=2, lineType=cv2.LINE_4)

    return image, bbox_image, match_image


def annotate_screen(screen: Screen) -> Image.Image:
    # Pad image to accomodate annotations
    x_pad, y_pad = 50, 50
    image = copy.deepcopy(screen.image)
    image = cv2.copyMakeBorder(image, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Check if UI is dark or light and assign text color
    gray = cv2.cvtColor(screen.image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (100, 100))
    avg_brightness = np.mean(gray_resized)
    text_color = (0, 255, 0) if avg_brightness < 128 else (255, 0, 0)
        
    # Mask out regions occupied by widgets in the image
    h, w, _ = screen.image.shape
    mask = np.zeros(shape=(h + 2*y_pad, w + 2*x_pad))
    for widget in screen.widgets.values():
        x_min, y_min, x_max, y_max = widget.bbox
        mask[y_pad+y_min:y_pad+y_max, x_pad+x_min:x_pad+x_max] = 1

    # For each widget, find an empty side to annotate, otherwise annotate center
    font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
    for id, widget in screen.widgets.items():
        id = str(id)
        margin = 40
        h, w, _ = image.shape
        x_min, y_min, x_max, y_max = widget.bbox
        x_min, x_max = x_min + x_pad, x_max + x_pad
        y_min, y_max = y_min + y_pad, y_max + y_pad
        text_size = cv2.getTextSize(id, font, font_scale, font_thickness)[0]

        # up
        if np.sum(mask[y_min-margin:y_min, x_min:x_max]) == 0:
            text_x = x_min
            text_y = y_min
        
        # left
        elif np.sum(mask[y_min:y_max, x_min-margin:x_min]) == 0:
            text_x = x_min - text_size[0]
            text_y = y_min + text_size[1]

        # down
        elif np.sum(mask[y_max:y_max+margin, x_min:x_max]) == 0:
            text_x = x_min
            text_y = y_max + text_size[1]

        # right
        elif np.sum(mask[y_min:y_max, x_max:x_max+margin]) == 0:
            text_x = x_max
            text_y = y_min + text_size[1]

        # center
        # else:
        #     rect_center_x = (x_min + x_max) // 2
        #     rect_center_y = (y_min + y_max) // 2
        #     text_x = rect_center_x - text_size[0] // 2
        #     text_y = rect_center_y + text_size[1] // 2
        else:
            text_x = x_min
            text_y = y_min + text_size[1]
            
        cv2.putText(image, id, (text_x, text_y), font, font_scale, text_color, font_thickness)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


def convert_process_to_json(path: str):
    process_actions = []

    with open(f"{path}/implementation/process.txt", 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            screen_actions = {
                "screen": f"{i + 1}.jpg",
                "actions": []
            }

            line = line.strip()
            actions = line.split("，")
            for action in actions:
                screen_actions["actions"].append({
                    "action": action,
                    "bounds": []
                })

            process_actions.append(screen_actions)
    
    # Convert list of actions to JSON
    with open(f"{path}/implementation/process.json", "w") as f: json.dump(process_actions, f, ensure_ascii=False, indent=2)


def check_action(true_action: dict, action_name: str, action) -> bool:
    def _overlap(box1: Iterable[int], box2: Iterable[int]) -> bool:
        # Unpack the coordinates of the boxes
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Check for overlap using the condition
        return (x1_min <= x2_max and x2_min <= x1_max and
                y1_min <= y2_max and y2_min <= y1_max)
    
    if action_name != true_action["action"]: 
        print("\t\t[x] Incorrect action name")
        return False
    
    bounds: list = action()
    true_bounds: list = [true_action.get("bounds")] if true_action.get("bounds") is not None else []

    if len(bounds) != len(true_bounds):
        print(bounds)
        print(true_bounds)
        print("\t\t[x] Incorrect bboxes")
        print(action_name, true_action)
        return False
    
    if not all([_overlap(a, b) for a, b in zip(bounds, true_bounds)]):
        print("\t\t[x] Incorrect overlap", bounds, true_bounds)
        return False
    
    true_direction = true_action.get("direction")
    if true_direction and action.args[-1] != true_direction:
        return False
    
    return True