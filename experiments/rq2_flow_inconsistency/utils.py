import os
import re
from copy import deepcopy
from functools import partial
from timeit import default_timer as timer
from typing import Callable, Iterable

import cv2
import numpy as np
from actions import Automator, Step, Translator
from PIL import Image

from experiments.common.utils import visualize_inconsistencies
from guipilot.agent import Agent
from guipilot.checker import ScreenChecker
from guipilot.entities import Screen
from guipilot.matcher import WidgetMatcher


def get_mock_screen(process_path: str, step: Step) -> Screen:
    mock_image_path = os.path.join(process_path, step.screenshot)
    mock_image: np.ndarray = cv2.imread(mock_image_path)
    mock_screen = Screen(mock_image)
    mock_screen.detect()
    mock_screen.ocr()
    return mock_screen


def get_real_screen(automator: Automator) -> Screen:
    real_image = automator.device.screenshot(format="opencv")
    real_screen = Screen(real_image)
    real_screen.detect()
    real_screen.ocr()
    return real_screen


def get_action_completion(
    agent: Agent, screen: Screen, step: Step
) -> tuple[tuple, list[str], list[Callable]]:
    description = step.description
    image = annotate_screen(screen)

    base_path = os.path.abspath(os.path.dirname(__file__))
    prompt_path = os.path.join(base_path, "action_completion.user.prompt")
    prompt = open(prompt_path).read()
    prompt = prompt.replace("{ACTION_DESCRIPTION}", description)

    response = agent(prompt, [image])

    print("[VLM]\n", response)

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

    visualize = (image, response)
    return visualize, action_names, actions


def get_scores(
    mock: Screen, real: Screen, matcher: WidgetMatcher, checker: ScreenChecker
) -> tuple[np.ndarray, tuple, tuple]:
    """Get consistency scores for the current (real) screen as compared to mock screen.

    Args:
        mock, real: The screens to check against.
        matcher: An algorithm that pairs matching widgets on both screens.
        checker: An algorithm that checks the consistency of paired widgets.

    Returns:
        tuple[float]: A tuple of different calculated scores.
        tuple[float]: A tuple of time taken for calculating scores.
    """
    start_time = timer()
    pairs, scores, _ = matcher.match(mock, real)
    score1 = sum(scores) / len(real.widgets)
    time1 = timer() - start_time

    # penalize unpaired mock widgets
    start_time = timer()
    inconsistencies, _ = checker.check(mock, real, pairs)
    unpaired_mock = {x[0] for x in inconsistencies if x[1] is None}
    unpaired_mock = len(unpaired_mock) / len(mock.widgets)
    score2 = score1 - unpaired_mock
    time2 = time1 + (timer() - start_time)

    # penalize unpaired real widgets
    start_time = timer()
    unpaired_real = {x[1] for x in inconsistencies if x[0] is None}
    unpaired_real = len(unpaired_real) / len(real.widgets)
    score3 = score1 - unpaired_real
    time3 = time1 + (timer() - start_time)

    visualize = visualize_inconsistencies(mock, real, pairs, inconsistencies)

    return visualize, (score1, score2, score3), (time1, time2, time3)


def execute_action(automator: Automator, step: Step) -> float:
    start_time = timer()
    params: dict = deepcopy(step.params)
    for _, value in params.items():
        if isinstance(value, dict):
            value.pop("bounds", None)

    action = getattr(automator, step.action)
    action(**params)
    return timer() - start_time


def annotate_screen(screen: Screen) -> Image.Image:
    # Pad image to accomodate annotations
    x_pad, y_pad = 50, 50
    image = deepcopy(screen.image)
    image = cv2.copyMakeBorder(
        image, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    # Check if UI is dark or light and assign text color
    gray = cv2.cvtColor(screen.image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (100, 100))
    avg_brightness = np.mean(gray_resized)
    text_color = (0, 255, 0) if avg_brightness < 128 else (255, 0, 0)

    # Mask out regions occupied by widgets in the image
    h, w, _ = screen.image.shape
    mask = np.zeros(shape=(h + 2 * y_pad, w + 2 * x_pad))
    for widget in screen.widgets.values():
        x_min, y_min, x_max, y_max = widget.bbox
        mask[y_pad + y_min : y_pad + y_max, x_pad + x_min : x_pad + x_max] = 1

    # For each widget, find an empty side to annotate, otherwise annotate center
    font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    for id, widget in screen.widgets.items():
        id = str(id)
        margin = 40
        h, w, _ = image.shape
        x_min, y_min, x_max, y_max = widget.bbox
        x_min, x_max = x_min + x_pad, x_max + x_pad
        y_min, y_max = y_min + y_pad, y_max + y_pad
        text_size = cv2.getTextSize(id, font, font_scale, font_thickness)[0]

        # up
        if np.sum(mask[y_min - margin : y_min, x_min:x_max]) == 0:
            text_x = x_min
            text_y = y_min

        # left
        elif np.sum(mask[y_min:y_max, x_min - margin : x_min]) == 0:
            text_x = x_min - text_size[0]
            text_y = y_min + text_size[1]

        # down
        elif np.sum(mask[y_max : y_max + margin, x_min:x_max]) == 0:
            text_x = x_min
            text_y = y_max + text_size[1]

        # right
        elif np.sum(mask[y_min:y_max, x_max : x_max + margin]) == 0:
            text_x = x_max
            text_y = y_min + text_size[1]

        # center
        else:
            rect_center_x = (x_min + x_max) // 2
            rect_center_y = (y_min + y_max) // 2
            text_x = rect_center_x - text_size[0] // 2
            text_y = rect_center_y + text_size[1] // 2

        cv2.putText(
            image, id, (text_x, text_y), font, font_scale, text_color, font_thickness
        )
        cv2.rectangle(
            image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2
        )

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


def check_overlap(box1: Iterable[int], box2: Iterable[int]) -> bool:
    # Unpack the coordinates of the boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Check for overlap using the condition
    return (
        x1_min <= x2_max and x2_min <= x1_max and y1_min <= y2_max and y2_min <= y1_max
    )
