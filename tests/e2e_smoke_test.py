import csv
import glob
import os
import random
import warnings
from copy import deepcopy

from experiments.rq1_screen_inconsistency.mutate import insert_row
from experiments.rq1_screen_inconsistency.utils import (
    filter_overlap_predictions,
    load_screen,
)
from guipilot.checker.gvt import GVT as GVTChecker
from guipilot.matcher.gvt import GVT as GVTMatcher


def smoke_test():
    warnings.filterwarnings("ignore")
    random.seed(42)

    # Use a small subset of the dataset
    dataset_path = os.getenv("DATASET_PATH", "./datasets/new")
    image_paths = glob.glob(os.path.join(dataset_path, "Adobe", "1.jpg"))

    if not image_paths:
        print("No images found for smoke test, skipping.")
        return

    image_path = image_paths[0]
    print(f"Running smoke test on: {image_path}")

    try:
        # Load screen and its widgets from JSON
        screen1 = load_screen(image_path)

        # In a real scenario, we might call screen1.ocr() or detect()
        # but here we use ground truth widgets from JSON.

        screen2 = deepcopy(screen1)
        # Apply a simple mutation
        screen2, y_true = insert_row(screen2, 0.05)

        matcher = GVTMatcher(threshold=screen1.image.shape[0] / 8)
        checker = GVTChecker()

        pairs, _, _ = matcher.match(screen1, screen2)
        y_pred, _ = checker.check(screen1, screen2, pairs)

        # Post-processing
        y_pred = filter_overlap_predictions(y_pred, y_true, None, screen2)

        # Write to evaluation.csv
        with open("evaluation.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "mutation", "matcher", "checker", "tp", "fp", "fn"])
            # Simplified metrics for smoke test
            a = set([(x[0], x[1]) for x in y_pred])
            b = set([(x[0], x[1]) for x in y_true])
            tp = len(a.intersection(b))
            fp = len(a.difference(b))
            fn = len(b.difference(a))
            writer.writerow([image_path, "insert_row", "gvt", "gvt", tp, fp, fn])

        print("Smoke test completed successfully. evaluation.csv generated.")

    except Exception as e:
        print(f"Smoke test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    smoke_test()
