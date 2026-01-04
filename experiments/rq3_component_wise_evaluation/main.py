import argparse
import csv
import os
import random
import sys
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

# Add project root to path for cross-experiment imports (experiments.rq1_screen_inconsistency)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv


if TYPE_CHECKING:
    from guipilot.checker import ScreenChecker
    from guipilot.entities import Screen
    from guipilot.matcher import WidgetMatcher


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    matcher_factory: Callable[["Screen"], "WidgetMatcher"]
    checker_factory: Callable[[], "ScreenChecker"]
    enable_ocr: bool = True
    enable_postprocess: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Component-wise evaluation for GUIPilot screen inconsistency detection."
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
        "--pipelines",
        default="guipilot_full,guipilot_no_postprocess,gvt_matcher",
        help="Comma-separated list of pipeline presets to run. "
        "Choices: guipilot_full, guipilot_no_postprocess, guipilot_no_ocr, gvt_matcher.",
    )
    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Skip generating visualization images and logs to speed up smoke tests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (used by mutation operators).",
    )
    return parser.parse_args()


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    dataset_candidate = args.dataset or os.getenv("DATASET_PATH")
    if not dataset_candidate:
        raise RuntimeError("Dataset path not provided. Use --dataset or set DATASET_PATH.")
    dataset_root = Path(dataset_candidate).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")
    os.environ["DATASET_PATH"] = str(dataset_root)
    return dataset_root


def get_pipeline_registry() -> dict[str, Callable[[], PipelineConfig]]:
    from guipilot.matcher import GUIPilotV2 as GUIPilotMatcher, GVT as GVTMatcher
    from guipilot.checker import GVT as GVTChecker

    return {
        "guipilot_full": lambda: PipelineConfig(
            name="guipilot_full",
            matcher_factory=lambda _screen: GUIPilotMatcher(),
            checker_factory=lambda: GVTChecker(),
            enable_ocr=True,
            enable_postprocess=True,
        ),
        "guipilot_no_postprocess": lambda: PipelineConfig(
            name="guipilot_no_postprocess",
            matcher_factory=lambda _screen: GUIPilotMatcher(),
            checker_factory=lambda: GVTChecker(),
            enable_ocr=True,
            enable_postprocess=False,
        ),
        "guipilot_no_ocr": lambda: PipelineConfig(
            name="guipilot_no_ocr",
            matcher_factory=lambda _screen: GUIPilotMatcher(),
            checker_factory=lambda: GVTChecker(),
            enable_ocr=False,
            enable_postprocess=True,
        ),
        "gvt_matcher": lambda: PipelineConfig(
            name="gvt_matcher",
            matcher_factory=lambda screen: GVTMatcher(screen.image.shape[0] / 8),
            checker_factory=lambda: GVTChecker(),
            enable_ocr=True,
            enable_postprocess=True,
        ),
    }


def ensure_valid_pipelines(requested: Iterable[str]) -> list[PipelineConfig]:
    registry = get_pipeline_registry()
    pipelines: list[PipelineConfig] = []
    for name in requested:
        if name not in registry:
            valid = ", ".join(sorted(registry.keys()))
            raise ValueError(f"Unknown pipeline preset '{name}'. Available: {valid}")
        pipelines.append(registry[name]())
    return pipelines


def metrics(y_pred: set, y_true: set) -> tuple[int, int, int, int]:
    """Calculate classification metrics shared with RQ1 evaluation."""
    a = set([(x[0], x[1]) for x in y_pred])
    b = set([(x[0], x[1]) for x in y_true])
    cls_tp = len(y_pred.intersection(y_true))
    tp = len(a.intersection(b))
    fn = len(b.difference(a))
    fp = len(a.difference(b))
    return cls_tp, tp, fp, fn


def evaluate_pipeline(
    pipeline: PipelineConfig,
    image_paths: list[Path],
    output_root: Path,
    skip_visualize: bool,
) -> dict[str, float]:
    from experiments.rq1_screen_inconsistency.utils import (
        load_screen,
        visualize_inconsistencies,
        filter_swapped_predictions,
        filter_overlap_predictions,
        filter_color,
        filter_text,
    )
    from experiments.rq1_screen_inconsistency.mutate import (
        insert_row,
        delete_row,
        swap_widgets,
        change_widgets_text,
        change_widgets_color,
    )
    mutations = {
        "insert_row": insert_row,
        "delete_row": delete_row,
        "swap_widgets": swap_widgets,
        "change_widgets_text": change_widgets_text,
        "change_widgets_color": change_widgets_color,
    }

    postprocessing = {
        "insert_row": lambda y_pred, y_true, s1, s2: filter_overlap_predictions(y_pred, y_true, None, s2),
        "delete_row": lambda y_pred, y_true, s1, s2: filter_overlap_predictions(y_pred, y_true, s1, None),
        "swap_widgets": lambda y_pred, y_true, s1, s2: filter_swapped_predictions(y_pred, y_true, s1, s2),
        "change_widgets_text": lambda y_pred, y_true, s1, s2: filter_color(y_pred, y_true, s1, None),
        "change_widgets_color": lambda y_pred, y_true, s1, s2: filter_text(y_pred, y_true, s1, None),
    }

    pipeline_dir = output_root / pipeline.name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    visualize_root = pipeline_dir / "visualize"

    evaluation_csv_path = pipeline_dir / "evaluation.csv"
    with evaluation_csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "pipeline",
                "image",
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

        aggregates = defaultdict(int)
        total_match_time = 0.0
        total_check_time = 0.0
        evaluation_count = 0

        checker: ScreenChecker = pipeline.checker_factory()

        for mutation_name, mutate in mutations.items():
            for image_path in image_paths:
                try:
                    screen1: Screen = load_screen(str(image_path))
                    if pipeline.enable_ocr:
                        screen1.ocr()
                    screen2 = deepcopy(screen1)
                    screen2, y_true = mutate(screen2, 0.05)
                    if pipeline.enable_ocr:
                        screen2.ocr()
                except Exception:
                    warnings.warn(f"[pipeline={pipeline.name}] Failed during mutation for {image_path}", RuntimeWarning)
                    continue

                matcher = pipeline.matcher_factory(screen1)
                try:
                    pairs, _, match_time = matcher.match(screen1, screen2)
                    y_pred, check_time = checker.check(screen1, screen2, pairs)
                except Exception:
                    warnings.warn(f"[pipeline={pipeline.name}] Failed during matching/checking for {image_path}", RuntimeWarning)
                    continue

                if pipeline.enable_postprocess:
                    y_pred_filtered = postprocessing[mutation_name](y_pred, y_true, screen1, screen2)
                else:
                    y_pred_filtered = y_pred

                cls_tp, tp, fp, fn = metrics(y_pred_filtered, y_true)

                if not skip_visualize:
                    parent_dir = image_path.parent.name
                    relative_path = f"{mutation_name}/{parent_dir}"
                    visualize_inconsistencies(
                        screen1,
                        screen2,
                        pairs,
                        y_pred_filtered,
                        relative_path,
                        image_path.stem,
                        output_root=visualize_root,
                    )

                writer.writerow(
                    [
                        pipeline.name,
                        str(image_path),
                        mutation_name,
                        matcher.__class__.__name__,
                        checker.__class__.__name__,
                        cls_tp,
                        tp,
                        fp,
                        fn,
                        match_time,
                        check_time,
                    ]
                )

                aggregates["cls_tp"] += cls_tp
                aggregates["tp"] += tp
                aggregates["fp"] += fp
                aggregates["fn"] += fn
                total_match_time += match_time
                total_check_time += check_time
                evaluation_count += 1

        precision = aggregates["tp"] / (aggregates["tp"] + aggregates["fp"]) if (aggregates["tp"] + aggregates["fp"]) else 0.0
        recall = aggregates["tp"] / (aggregates["tp"] + aggregates["fn"]) if (aggregates["tp"] + aggregates["fn"]) else 0.0
        cls_precision = (
            aggregates["cls_tp"] / aggregates["tp"] if aggregates["tp"] else 0.0
        )

        avg_match_time = total_match_time / evaluation_count if evaluation_count else 0.0
        avg_check_time = total_check_time / evaluation_count if evaluation_count else 0.0

        summary = {
            "pipeline": pipeline.name,
            "cls_tp": aggregates["cls_tp"],
            "tp": aggregates["tp"],
            "fp": aggregates["fp"],
            "fn": aggregates["fn"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "cls_precision": round(cls_precision, 4),
            "avg_match_time_ms": round(avg_match_time, 2),
            "avg_check_time_ms": round(avg_check_time, 2),
            "evaluations": evaluation_count,
        }

    return summary


def write_summary(output_root: Path, summaries: list[dict[str, float]]) -> None:
    summary_csv_path = output_root / "summary.csv"
    if not summaries:
        return
    fieldnames = list(summaries[0].keys())
    with summary_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    load_dotenv()
    warnings.filterwarnings("ignore")
    random.seed(args.seed)

    dataset_root = resolve_dataset_path(args)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_paths: list[Path] = [
        path
        for path in sorted(dataset_root.rglob("*.jpg"))
        if path.stem.isdigit()
    ]
    if args.limit is not None:
        all_paths = all_paths[: args.limit]
    if not all_paths:
        raise RuntimeError(f"No valid *.jpg files found under dataset: {dataset_root}")

    requested_pipelines = [name.strip() for name in args.pipelines.split(",") if name.strip()]
    pipelines = ensure_valid_pipelines(requested_pipelines)

    print(f"[RQ3] Evaluating {len(all_paths)} screens with pipelines: {', '.join([p.name for p in pipelines])}")

    summaries: list[dict[str, float]] = []
    for pipeline in pipelines:
        summary = evaluate_pipeline(
            pipeline,
            all_paths,
            output_root,
            skip_visualize=args.skip_visualize,
        )
        summaries.append(summary)
        precision = summary["precision"]
        recall = summary["recall"]
        cls_precision = summary["cls_precision"]
        print(
            f"[{pipeline.name}] "
            f"cls_precision={cls_precision:.2f} precision={precision:.2f} recall={recall:.2f} "
            f"evaluations={summary['evaluations']}"
        )

    write_summary(output_root, summaries)
    print(f"[RQ3] Summaries written to {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()

