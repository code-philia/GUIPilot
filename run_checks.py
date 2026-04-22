"""Run GUIPilot screen-consistency checks on mock/real image pairs."""

import argparse
import os
import re
import csv
import glob
import warnings

# Must be set before any paddle/paddleocr import to avoid the OneDNN PIR bug on CPU.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

import cv2
import guipilot.entities.screen as _screen_module
from guipilot.models import OCR, Detector
from guipilot.entities import Screen, Inconsistency
from guipilot.matcher import GUIPilotV2 as Matcher
from guipilot.checker import GVT as Checker
from guipilot.visualize import visualize_inconsistencies
from guipilot.postprocess import postprocess

warnings.filterwarnings("ignore")

DETECTOR_DIR = os.path.join(os.path.dirname(__file__), "guipilot", "models", "detector")
MODEL_PATH   = os.path.join(DETECTOR_DIR, "best.pt")
HF_MODEL_URL = "https://huggingface.co/code-philia/GUIPilot/resolve/main/widget_detector.pt"


def convert_inconsistencies(inconsistencies: set) -> set:
    result = set()
    for item in inconsistencies:
        id1, id2 = item[0], item[1]
        itype = item[2] if len(item) > 2 else None

        if id1 is None:
            result.add(("insert", id2))
        elif id2 is None:
            result.add(("delete", id1))
        elif itype == Inconsistency.COLOR:
            result.add(("substitute.color", id1, id2))
        elif itype == Inconsistency.TEXT:
            result.add(("substitute.text", id1, id2))
        elif itype == Inconsistency.BBOX and (id2, id1, itype) in inconsistencies:
            result.add(("substitute.swap", id1, id2))
        elif itype == Inconsistency.BBOX:
            result.add(("substitute.bbox", id1, id2))
    return result


def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading YOLO model weights to {MODEL_PATH} ...")
    import urllib.request
    os.makedirs(DETECTOR_DIR, exist_ok=True)
    urllib.request.urlretrieve(HF_MODEL_URL, MODEL_PATH)
    print("Download complete.")


def patch_screen_module():
    """Replace the module-level OCR/Detector instances with local (no-service) ones."""
    _screen_module.ocr = OCR(service_url=None)
    _screen_module.detector = Detector(service_url=None)


def find_pairs(directory: str) -> list[tuple[str, str]]:
    mocks = sorted(glob.glob(os.path.join(directory, "*_mock.*")))
    pairs = []
    for mock_path in mocks:
        stem = re.sub(r"_mock\.[^.]+$", "", mock_path)
        ext = os.path.splitext(mock_path)[1]
        real_path = stem + "_real" + ext
        if os.path.exists(real_path):
            pairs.append((mock_path, real_path))
        else:
            print(f"[warn] no matching real image for {os.path.basename(mock_path)}")
    return pairs


def load_screen(image_path: str) -> Screen:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    return Screen(image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GUIPilot screen-consistency checks on mock/real image pairs."
    )
    parser.add_argument(
        "--input", required=True, metavar="DIR",
        help="Directory containing *_mock.<ext> / *_real.<ext> image pairs.",
    )
    parser.add_argument(
        "--output", required=True, metavar="DIR",
        help="Directory where results.csv, per-pair .txt reports, and visualizations are saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir    = os.path.abspath(args.input)
    output_dir    = os.path.abspath(args.output)
    results_csv   = os.path.join(output_dir, "results.csv")
    visualize_dir = os.path.join(output_dir, "visualizations")

    if not os.path.isdir(images_dir):
        raise SystemExit(f"Input directory not found: {images_dir}")
    os.makedirs(output_dir, exist_ok=True)

    download_model()
    patch_screen_module()

    matcher = Matcher()
    checker = Checker()

    pairs = find_pairs(images_dir)
    if not pairs:
        print(f"No mock/real pairs found in {images_dir}")
        return

    print(f"Found {len(pairs)} pair(s) in {images_dir}\n")

    with open(results_csv, "w", newline="") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow([
            "pair", "mock_widgets", "real_widgets",
            "matched_pairs", "match_time_ms", "check_time_ms",
            "mock_widget_id", "real_widget_id", "inconsistency_type",
        ])

        for mock_path, real_path in pairs:
            name = os.path.basename(mock_path).replace("_mock", "").rsplit(".", 1)[0]
            print(f"=== Pair: {name} ===")

            try:
                mock_screen = load_screen(mock_path)
                real_screen = load_screen(real_path)

                mock_screen.detect()
                real_screen.detect()
                mock_screen.ocr()
                real_screen.ocr()

                n_mock = len(mock_screen.widgets)
                n_real = len(real_screen.widgets)
                print(f"  widgets — mock: {n_mock}, real: {n_real}")

                matched_pairs, _, match_ms = matcher.match(mock_screen, real_screen)
                raw_pred, check_ms = checker.check(mock_screen, real_screen, matched_pairs)
                inconsistencies = postprocess(raw_pred, mock_screen, real_screen)

                print(f"  matched {len(matched_pairs)} pair(s) [{match_ms}ms match, {check_ms}ms check]")

                # --- Text report (console + .txt file) ---
                report_lines = [
                    "\n--matched--",
                    repr(matched_pairs),
                    "\n--inconsistencies--",
                    f"y_pred: {inconsistencies}",
                    "\n--edit_distance--",
                    f"y_pred: {convert_inconsistencies(inconsistencies)}",
                    "\n--raw_pred--",
                    repr(raw_pred),
                ]
                report = "\n".join(report_lines)
                print(report)

                os.makedirs(visualize_dir, exist_ok=True)
                txt_path = os.path.join(visualize_dir, f"{name}.txt")
                with open(txt_path, "w") as txt_f:
                    txt_f.write(report + "\n")

                # --- Visualization ---
                vis_path = os.path.join(visualize_dir, f"{name}.jpg")
                visualize_inconsistencies(
                    mock_screen, real_screen, matched_pairs, inconsistencies, vis_path
                )
                print(f"  saved → {vis_path}")

                # --- CSV ---
                if inconsistencies:
                    for item in inconsistencies:
                        id1, id2 = item[0], item[1]
                        itype = item[2].name if len(item) > 2 else None
                        writer.writerow([
                            name, n_mock, n_real,
                            len(matched_pairs), match_ms, check_ms,
                            id1, id2, itype,
                        ])
                else:
                    writer.writerow([
                        name, n_mock, n_real,
                        len(matched_pairs), match_ms, check_ms,
                        None, None, None,
                    ])

            except Exception:
                import traceback
                print("  ERROR:")
                traceback.print_exc()
                writer.writerow([name, None, None, None, None, None, None, None, "ERROR"])

            print()

    print(f"Results saved to {results_csv}")


if __name__ == "__main__":
    main()
