import csv
import glob
import os
import random
import warnings
from copy import deepcopy
from typing import Callable

from dotenv import load_dotenv
from mutate import (
    change_widgets_color,
    change_widgets_text,
    delete_row,
    insert_row,
    swap_widgets,
)
from utils import (
    convert_inconsistencies,
    filter_color,
    filter_overlap_predictions,
    filter_swapped_predictions,
    filter_text,
    load_screen,
    visualize_inconsistencies,
)

from guipilot.checker import GVT as GVTChecker
from guipilot.checker import ScreenChecker
from guipilot.entities import Screen
from guipilot.matcher import GVT as GVTMatcher
from guipilot.matcher import GUIPilotV2 as GUIPilotMatcher
from guipilot.matcher import WidgetMatcher


def metrics(y_pred: set, y_true: set) -> tuple[int, int, int, int]:
    """Calculate
    1. cls_tp: no. of inconsistencies reported (correct pair & type)
    2. tp: no. of inconsistencies reported (correct pair)
    3. fn: no. of inconsistencies not reported
    4. fp: no. of inconsistencies falsely reported
    """
    a = {(x[0], x[1]) for x in y_pred}
    b = {(x[0], x[1]) for x in y_true}
    cls_tp = len(y_pred.intersection(y_true))
    tp = len(a.intersection(b))
    fn = len(b.difference(a))
    fp = len(a.difference(b))
    return cls_tp, tp, fp, fn


if __name__ == "__main__":
    load_dotenv()
    warnings.filterwarnings("ignore")
    random.seed(42)

    dataset_path = os.getenv("DATASET_PATH")
    all_paths: list[str] = []
    for app_path in glob.glob(os.path.join(dataset_path, "**", "*")):
        image_paths = glob.glob(f"{app_path}/*.jpg")
        image_paths = [
            x for x in image_paths if x.split("/")[-1].replace(".jpg", "").isdigit()
        ]
        image_paths.sort(key=lambda path: int(path.split("/")[-1].replace(".jpg", "")))
        all_paths += image_paths

    mutations = {
        "insert_row": insert_row,
        "delete_row": delete_row,
        "swap_widgets": swap_widgets,
        "change_widgets_text": change_widgets_text,
        "change_widgets_color": change_widgets_color,
    }

    postprocessing = {
        "insert_row": lambda y_pred, y_true, s1, s2: filter_overlap_predictions(
            y_pred, y_true, None, s2
        ),
        "delete_row": lambda y_pred, y_true, s1, s2: filter_overlap_predictions(
            y_pred, y_true, s1, None
        ),
        "swap_widgets": lambda y_pred, y_true, s1, s2: filter_swapped_predictions(
            y_pred, y_true, s1, s2
        ),
        "change_widgets_text": lambda y_pred, y_true, s1, s2: filter_color(
            y_pred, y_true, s1, None
        ),
        "change_widgets_color": lambda y_pred, y_true, s1, s2: filter_text(
            y_pred, y_true, s1, None
        ),
    }

    matchers: dict[str, Callable] = {
        "gvt": lambda screen: GVTMatcher(screen.image.shape[0] / 8),
        "guipilot": lambda screen: GUIPilotMatcher(),
    }

    checkers: dict[str, ScreenChecker] = {"gvt": GVTChecker()}

    writer = csv.writer(open("./evaluation.csv", "w"))
    writer.writerow(
        [
            "id",
            "mutation",
            "matcher",
            "checker",
            "cls_tp",
            "tp",
            "fp",
            "fn",
            "match_time",
            "check_time",
        ]
    )

    # Iterate through all screens in public app dataset
    for mutation_name, mutate in mutations.items():
        for image_path in all_paths:
            try:
                screen1: Screen = load_screen(image_path)
                screen1.ocr()
                screen2 = deepcopy(screen1)
                screen2, y_true = mutate(screen2, 0.05)
                screen2.ocr()
            except Exception:
                import traceback

                print(traceback.format_exc())
                print("error during mutation, skipped")
                continue

            for matcher_name, init_matcher in matchers.items():
                for checker_name, checker in checkers.items():
                    try:
                        matcher: WidgetMatcher = init_matcher(screen1)
                        pairs, _, match_time = matcher.match(screen1, screen2)
                        y_pred, check_time = checker.check(screen1, screen2, pairs)

                        # Filter predictions for metrics
                        y_pred_raw = y_pred
                        y_pred = postprocessing[mutation_name](
                            y_pred, y_true, screen1, screen2
                        )

                        # Visualize
                        _path = image_path.split("/")[-2]
                        _path = f"{matcher_name}_{checker_name}/{mutation_name}/{_path}"
                        _filename = image_path.split("/")[-1].replace(".jpg", "")
                        # visualize_inconsistencies(
                        #     screen1, screen2, pairs, y_pred, _path, _filename
                        # )
                        save_path = f"./visualize/{_path}/{_filename}.jpg"
                        visualize_inconsistencies(
                            screen1, screen2, pairs, y_pred, save_path=save_path
                        )
                        with open(f"./visualize/{_path}/{_filename}.txt", "w") as f:
                            f.writelines(
                                [
                                    "\n--matched--\n",
                                    f"{pairs}\n",
                                    "\n--inconsistencies--\n",
                                    f"y_pred: {y_pred}\n",
                                    f"y_true: {y_true}\n\n--edit_distance--\n",
                                    f"y_pred: {convert_inconsistencies(y_pred)}\n",
                                    f"y_true: {convert_inconsistencies(y_true)}\n",
                                    "\n--raw_pred--\n",
                                    f"{y_pred_raw}",
                                ]
                            )

                        cls_tp, tp, fp, fn = metrics(y_pred, y_true)

                    except Exception:
                        import traceback

                        print(traceback.format_exc())
                        print("error during consistency checking, skipped")
                        continue

                    try:
                        cls_precision = round(cls_tp / tp, 2)
                    except:
                        cls_precision = 0.0

                    try:
                        precision = round(tp / (tp + fp), 2)
                    except:
                        precision = 0.0

                    try:
                        recall = round(tp / (tp + fn), 2)
                    except:
                        recall = 0.0

                    print(
                        f"{mutation_name} |",
                        f"{image_path.split('/')[-2]}/{image_path.split('/')[-1]} |",
                        "{:<10}".format(matcher_name),
                        "{:<10}".format(checker_name),
                        "|",
                        f"{cls_precision} {precision} {recall}",
                    )

                    writer.writerow(
                        [
                            image_path,
                            mutation_name,
                            matcher_name,
                            checker_name,
                            cls_tp,
                            tp,
                            fp,
                            fn,
                            match_time,
                            check_time,
                        ]
                    )
