import argparse
import os
import csv
import sys
import warnings
import random
from copy import deepcopy
from pathlib import Path
from typing import Callable

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add experiment directory to path for local imports (utils, mutate)
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from dotenv import load_dotenv

from utils import (
    load_screen,
    visualize_inconsistencies,
    convert_inconsistencies,
    filter_swapped_predictions,
    filter_overlap_predictions,
    filter_color,
    filter_text
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate screen inconsistency detection on a labeled dataset."
    )
    parser.add_argument(
        "--dataset",
        help="Path to the dataset root containing *.jpg/*.json pairs. "
        "Defaults to the DATASET_PATH environment variable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of screens to evaluate (after sorting).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to store evaluation results (CSV, visualization). Defaults to current directory.",
    )
    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Skip generating visualization images and logs to speed up smoke tests.",
    )
    return parser.parse_args()


def metrics(y_pred: set, y_true: set) -> tuple[int, int, int, int]:
    """Calculate
        1. cls_tp: no. of inconsistencies reported (correct pair & type)
        2. tp: no. of inconsistencies reported (correct pair)
        3. fn: no. of inconsistencies not reported
        4. fp: no. of inconsistencies falsely reported
    """
    a = set([(x[0], x[1]) for x in y_pred])
    b = set([(x[0], x[1]) for x in y_true])
    cls_tp = len(y_pred.intersection(y_true))
    tp = len(a.intersection(b))
    fn = len(b.difference(a))
    fp = len(a.difference(b))
    return cls_tp, tp, fp, fn


if __name__ == "__main__":
    args = parse_args()
    load_dotenv()
    warnings.filterwarnings("ignore")
    random.seed(42)

    dataset_candidate = args.dataset or os.getenv("DATASET_PATH")
    if not dataset_candidate:
        raise RuntimeError("Dataset path not provided. Use --dataset or set DATASET_PATH.")

    dataset_root = Path(dataset_candidate).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

    os.environ["DATASET_PATH"] = str(dataset_root)

    from guipilot.matcher import (
        WidgetMatcher,
        GUIPilotV2 as GUIPilotMatcher,
        GVT as GVTMatcher,
    )
    from guipilot.checker import (
        ScreenChecker,
        GVT as GVTChecker,
    )
    from guipilot.entities import Screen
    from mutate import (
        insert_row,
        delete_row,
        swap_widgets,
        change_widgets_text,
        change_widgets_color,
    )

    all_paths: list[Path] = [
        path
        for path in sorted(dataset_root.rglob("*.jpg"))
        if path.stem.isdigit()
    ]

    if args.limit is not None:
        all_paths = all_paths[: args.limit]

    mutations = {
        "insert_row": insert_row,
        "delete_row": delete_row,
        "swap_widgets": swap_widgets,
        "change_widgets_text": change_widgets_text,
        "change_widgets_color": change_widgets_color
    }

    postprocessing = {
        "insert_row": lambda y_pred, y_true, s1, s2: filter_overlap_predictions(y_pred, y_true, None, s2),
        "delete_row": lambda y_pred, y_true, s1, s2: filter_overlap_predictions(y_pred, y_true, s1, None),
        "swap_widgets": lambda y_pred, y_true, s1, s2: filter_swapped_predictions(y_pred, y_true, s1, s2),
        "change_widgets_text": lambda y_pred, y_true, s1, s2: filter_color(y_pred, y_true, s1, None),
        "change_widgets_color": lambda y_pred, y_true, s1, s2: filter_text(y_pred, y_true, s1, None)
    }

    matchers: dict[str, Callable] = {
        "gvt": lambda screen: GVTMatcher(screen.image.shape[0] / 8),
        "guipilot": lambda screen: GUIPilotMatcher()
    }

    checkers: dict[str, ScreenChecker] = {
        "gvt": GVTChecker()
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_csv_path = output_dir / "evaluation.csv"
    writer = csv.writer(open(evaluation_csv_path, "w", newline=""))
    writer.writerow(["id", "mutation", "matcher", "checker", "cls_tp", "tp", "fp", "fn", "match_time", "check_time"])
    visualize_root = output_dir / "visualize"
    
    # Iterate through all screens in dataset
    for mutation_name, mutate in mutations.items():
        for image_path in all_paths:

            try:
                image_path_str = str(image_path)
                screen1: Screen = load_screen(image_path_str)
                screen1.ocr()
                screen2 = deepcopy(screen1)
                screen2, y_true = mutate(screen2, 0.05)
                screen2.ocr()
            except Exception as e:
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
                        y_pred = postprocessing[mutation_name](y_pred, y_true, screen1, screen2)
                        
                        # Visualize
                        parent_dir = image_path.parent.name
                        relative_path = f"{matcher_name}_{checker_name}/{mutation_name}/{parent_dir}"
                        _filename = image_path.stem
                        if not args.skip_visualize:
                            visualize_inconsistencies(
                                screen1,
                                screen2,
                                pairs,
                                y_pred,
                                relative_path,
                                _filename,
                                output_root=visualize_root,
                            )
                            log_dir = visualize_root / relative_path
                            log_dir.mkdir(parents=True, exist_ok=True)
                            with open(log_dir / f"{_filename}.txt", "w") as f:
                                f.writelines([
                                    f"\n--matched--\n",
                                    f"{pairs}\n",
                                    f"\n--inconsistencies--\n",
                                    f"y_pred: {y_pred}\n",
                                    f"y_true: {y_true}\n",
                                    f"\n--edit_distance--\n",
                                    f"y_pred: {convert_inconsistencies(y_pred)}\n",
                                    f"y_true: {convert_inconsistencies(y_true)}\n",
                                    f"\n--raw_pred--\n",
                                    f"{y_pred_raw}",
                                ])

                        cls_tp, tp, fp, fn = metrics(y_pred, y_true)
                        
                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        print("error during consistency checking, skipped")
                        continue

                    try: cls_precision = round(cls_tp / tp, 2)
                    except: cls_precision = 0.0

                    try: precision = round(tp / (tp + fp), 2)
                    except: precision = 0.0

                    try: recall = round(tp / (tp + fn), 2)
                    except: recall = 0.0

                    print(
                        f"{mutation_name} |",
                        f"{image_path.parent.name}/{image_path.name} |",
                        "{:<10}".format(matcher_name),
                        "{:<10}".format(checker_name),
                        "|",
                        f"{cls_precision} {precision} {recall}"
                    )

                    writer.writerow([
                        str(image_path), mutation_name, matcher_name, checker_name,
                        cls_tp, tp, fp, fn,
                        match_time, check_time
                    ])