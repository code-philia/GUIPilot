import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from dotenv import load_dotenv

# Add experiment directory to path for local imports (utils)
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from guipilot.agent import GPTAgent
from guipilot.matcher import GUIPilotV2 as GUIPilotMatcher
from guipilot.checker import GVT as GVTChecker
from utils import (
    get_screen,
    get_scores,
    get_report,
    get_action_completion,
    visualize,
    check_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RQ4 case study evaluation for mockup vs. implementation consistency."
    )
    parser.add_argument(
        "--dataset",
        help="Path to case study dataset root. Defaults to DATASET_PATH environment variable.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to store reports, images, and summary CSV. Defaults to ./output.",
    )
    parser.add_argument(
        "--process-limit",
        type=int,
        help="Limit the number of processes to evaluate.",
    )
    parser.add_argument(
        "--screen-limit",
        type=int,
        help="Limit the number of screens per process.",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Skip VLM agent inference (useful for CI smoke tests without API key).",
    )
    parser.add_argument(
        "--max-agent-trials",
        type=int,
        default=3,
        help="Maximum retries for the VLM agent when not skipping actions.",
    )
    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Skip writing visualization images to reduce runtime.",
    )
    parser.add_argument(
        "--include-branches",
        action="store_true",
        help="Include branch screens (filenames containing 'branch'). Default会忽略。",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="若筛选后无可用屏幕，则不报错，直接退出。",
    )
    return parser.parse_args()


def get_processes(dataset_root: Path) -> list[Path]:
    parent_dirs = set()
    for root, dirs, _ in os.walk(dataset_root):
        if "mockup" in dirs and "implementation" in dirs:
            parent_dirs.add(Path(root))
    return sorted(parent_dirs)


def ensure_dataset_path(args: argparse.Namespace, output_root: Path) -> Path:
    dataset_candidate = args.dataset or os.getenv("DATASET_PATH")
    if not dataset_candidate:
        raise RuntimeError("Dataset path not provided. Use --dataset or set DATASET_PATH.")

    dataset_root = Path(dataset_candidate).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

    os.environ["DATASET_PATH"] = str(dataset_root)
    return dataset_root


def create_demo_dataset(target_root: Path) -> None:
    """Generate a minimal synthetic dataset for smoke testing."""
    implementation_dir = target_root / "demo_app/process_1/implementation"
    mockup_dir = target_root / "demo_app/process_1/mockup"
    implementation_dir.mkdir(parents=True, exist_ok=True)
    mockup_dir.mkdir(parents=True, exist_ok=True)

    # Create simple placeholder images
    height, width = 1280, 720
    base_image = np.full((height, width, 3), 240, dtype=np.uint8)
    cv2.rectangle(base_image, (50, 200), (650, 400), (0, 0, 0), 3)
    cv2.putText(base_image, "Demo Screen", (80, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    real_image = base_image.copy()
    cv2.circle(real_image, (600, 250), 20, (0, 0, 255), -1)

    cv2.imwrite(str(implementation_dir / "1.jpg"), real_image)
    cv2.imwrite(str(mockup_dir / "1.png"), base_image)

    process = [
        {
            "screen": "1.jpg",
            "mock_actions": ["点击 Demo 按钮"],
            "actions": [
                {
                    "action": "click",
                    "bounds": [50, 200, 650, 400],
                }
            ],
        }
    ]

    with (implementation_dir / "process.json").open("w", encoding="utf-8") as fh:
        json.dump(process, fh, ensure_ascii=False, indent=2)


def resolve_agent(skip_agent: bool) -> GPTAgent | None:
    if skip_agent:
        return None
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_KEY is not set. Provide the key or use --skip-agent.")
    return GPTAgent(api_key=api_key)


def evaluate_case_study(
    dataset_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> list[dict]:
    matcher = GUIPilotMatcher()
    checker = GVTChecker()
    agent = resolve_agent(args.skip_agent)

    processes = get_processes(dataset_root)
    if args.process_limit is not None:
        processes = processes[: args.process_limit]

    records: list[dict] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for process_idx, process_path in enumerate(processes):
        implementation_path = process_path / "implementation"
        mockup_path = process_path / "mockup"
        json_path = implementation_path / "process.json"

        if not json_path.exists():
            continue

        with json_path.open(encoding="utf-8") as fh:
            steps: list[dict] = json.load(fh)

        if args.screen_limit is not None:
            steps = steps[: args.screen_limit]

        for screen_idx, step in enumerate(steps):
            screen_filename: str = step.get("screen", "")
            if not args.include_branches and "branch" in screen_filename:
                continue

            implementation_file = implementation_path / screen_filename
            mockup_file = mockup_path / screen_filename.replace(".jpg", ".png")
            if not implementation_file.exists() or not mockup_file.exists():
                continue

            tolerate_failure = args.allow_empty
            real_screen = get_screen(str(implementation_path), screen_filename, tolerate_failure=tolerate_failure)
            mock_screen = get_screen(str(mockup_path), screen_filename.replace(".jpg", ".png"), tolerate_failure=tolerate_failure)

            if real_screen is None or mock_screen is None:
                continue

            if len(real_screen.widgets) == 0 or len(mock_screen.widgets) == 0:
                continue

            pairs, score, match_time = get_scores(mock_screen, real_screen, matcher)
            inconsistencies, check_time = checker.check(mock_screen, real_screen, pairs)

            action_trials: list[str] = []
            action_status: str | bool = False
            agent_image = None

            mock_actions = step.get("mock_actions", [])
            true_actions = step.get("actions", [])

            if agent is None:
                action_status = "skipped"
            else:
                attempt = 0
                while not action_status and attempt < args.max_agent_trials:
                    attempt += 1
                    agent_image, action_names, actions_raw, actions = get_action_completion(agent, real_screen, mock_actions)
                    if len(actions) != len(true_actions):
                        action_trials.append(str(actions_raw))
                        continue

                    if not all(
                        check_action(true_action, action_name, action)
                        for true_action, action_name, action in zip(true_actions, action_names, actions)
                    ):
                        action_trials.append(str(actions_raw))
                        continue

                    action_trials.append(str(actions_raw))
                    action_status = True

            vis_image, bbox_image, match_image = visualize(mock_screen, real_screen, pairs, inconsistencies)

            process_output = output_root / f"process_{process_idx}" / f"screen_{screen_idx}"
            process_output.mkdir(parents=True, exist_ok=True)

            if not args.skip_visualize:
                cv2.imwrite(str(process_output / "image.jpg"), vis_image)
                cv2.imwrite(str(process_output / "image_bbox.jpg"), bbox_image)
                cv2.imwrite(str(process_output / "image_match.jpg"), match_image)
                if agent_image is not None:
                    cv2.imwrite(str(process_output / "image_agent.jpg"), agent_image)

            report = get_report(
                str(process_path),
                screen_idx,
                match_time,
                check_time,
                pairs,
                inconsistencies,
                action_status,
                action_trials,
            )
            with (process_output / "report.json").open("w", encoding="utf-8") as fh:
                fh.write(report)

            records.append(
                {
                    "process": process_path.name,
                    "screen_index": screen_idx,
                    "screen": screen_filename,
                    "score": score,
                    "inconsistency_count": len(inconsistencies),
                    "match_time_ms": match_time,
                    "check_time_ms": check_time,
                    "action_status": action_status,
                    "action_trials": ";".join(action_trials),
                }
            )

    return records


def write_summary(records: Iterable[dict], output_root: Path) -> None:
    records = list(records)
    if not records:
        return

    fieldnames = list(records[0].keys())
    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    load_dotenv()

    output_root = Path(args.output_dir).expanduser().resolve()
    dataset_root = ensure_dataset_path(args, output_root)

    records = evaluate_case_study(dataset_root, output_root, args)

    if not records and not args.allow_empty:
        raise RuntimeError("未找到可评估的屏幕。请检查数据集路径或放宽筛选条件。")

    write_summary(records, output_root)
    print(f"[RQ4] 评估完成，结果写入 {output_root}")


if __name__ == "__main__":
    main()