from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path

import cv2
from dotenv import load_dotenv

# Add experiment directory to path for local imports (actions, utils)
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from actions import Automator, Record, Translator
from utils import (
    get_action_completion,
    get_mock_screen,
    get_real_screen,
    get_replay_screen,
    get_scores,
    execute_action,
    check_overlap,
    annotate_screen,
)
from guipilot.agent import GPTAgent
from guipilot.entities import Screen
from guipilot.matcher import GUIPilotV2 as GUIPilotMatcher
from guipilot.checker import GVT as GVTChecker


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate flow inconsistency detection with optional offline replay mode.",
    )
    parser.add_argument(
        "--dataset",
        help="Path to dataset root containing process_* directories. Defaults to DATASET_PATH.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of processes to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/rq2",
        help="Directory to store evaluation outputs (visualizations, metrics).",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "replay"],
        default="interactive",
        help="Interactive mode uses connected device; replay mode reuses recorded screens from disk.",
    )
    parser.add_argument(
        "--replay-real-subdir",
        default="real",
        help="Sub-directory under each process_* folder containing real screens for replay mode.",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Skip the action completion agent stage (recommended for CI).",
    )
    parser.add_argument(
        "--process-pattern",
        default="process_*",
        help="Glob pattern for process folders inside the dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when selecting the inconsistent step.",
    )
    parser.add_argument(
        "--inconsistency-index",
        type=int,
        help="Force the inconsistent step index (0-based). Overrides random selection.",
    )
    parser.add_argument(
        "--results-file",
        default="results.csv",
        help="Filename for the aggregated metrics CSV inside the output directory.",
    )
    parser.add_argument(
        "--openai-key",
        help="Override OPENAI_KEY environment variable for the action completion agent.",
    )
    return parser.parse_args(argv)


def resolve_dataset_path(dataset_arg: str | None) -> Path:
    dataset_candidate = dataset_arg or os.getenv("DATASET_PATH")
    if not dataset_candidate:
        raise RuntimeError("Dataset path not provided. Use --dataset or set DATASET_PATH.")

    dataset_root = Path(dataset_candidate).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

    return dataset_root


def determine_inconsistency_index(
    record: Record,
    record_extra: dict[str, object],
    override_index: int | None,
) -> int:
    if override_index is not None:
        return override_index

    index_from_record = record_extra.get("inconsistency_index")
    if isinstance(index_from_record, int):
        return index_from_record

    if len(record.steps) <= 1:
        return -1

    return random.choice(list(range(0, len(record.steps) - 1)))


def ensure_agent(skip_agent: bool, openai_key: str | None) -> GPTAgent | None:
    if skip_agent:
        return None

    if not openai_key:
        raise RuntimeError(
            "OPENAI_KEY not provided. Use --openai-key, set the environment variable, "
            "or run with --skip-agent."
        )

    return GPTAgent(api_key=openai_key)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv()
    warnings.filterwarnings("ignore")
    random.seed(args.seed)

    dataset_root = resolve_dataset_path(args.dataset)
    process_paths = sorted(dataset_root.glob(args.process_pattern))
    if args.limit is not None:
        process_paths = process_paths[: args.limit]
    if not process_paths:
        raise RuntimeError(f"No process directories found under {dataset_root}.")

    matcher = GUIPilotMatcher()
    checker = GVTChecker()
    automator = Automator() if args.mode == "interactive" else None
    openai_key = args.openai_key or os.getenv("OPENAI_KEY")
    agent = ensure_agent(args.skip_agent, openai_key)

    results_dir = Path(args.output_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    visualize_root = results_dir / "visualize"
    visualize_root.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / args.results_file

    with open(report_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "id",
            "score1",
            "score2",
            "score3",
            "action_time",
            "time1",
            "time2",
            "time3",
            "ground_truth",
            "is_completed",
            "retries",
        ])

        for process_idx, process_path in enumerate(process_paths):
            record_path = process_path / "record.json"
            if not record_path.exists():
                raise FileNotFoundError(f"Record file not found: {record_path}")

            record_text = record_path.read_text()
            record = Record.model_validate_json(record_text)
            record_extra = json.loads(record_text)
            
            # Load VLM retry data from separate file if exists
            vlm_retry_path = process_path / "vlm_retry.json"
            vlm_retry_data = {}
            if vlm_retry_path.exists():
                vlm_retry_data = json.loads(vlm_retry_path.read_text())

            process_name = process_path.name
            package_name = record.package_name
            inconsistent_index = determine_inconsistency_index(
                record,
                record_extra,
                args.inconsistency_index,
            )
            print(f"[{process_idx+1}/{len(process_paths)}] {package_name}::{process_name} | inconsistent step = {inconsistent_index}")

            if args.mode == "interactive" and automator is not None:
                print("Launching app...")
                automator.launch(record.package_name, record.init_activity)
                input("[MANUAL] Align phone screen, then continue.")

            process_visualize_dir = visualize_root / f"{package_name}-{process_name}"
            process_visualize_dir.mkdir(parents=True, exist_ok=True)

            for step_index, step in enumerate(record.steps[:-1]):
                print(f"  - Step {step_index+1}/{len(record.steps[:-1])}: {step.description}")

                next_step = record.steps[step_index + 1]
                mock_screen: Screen = get_mock_screen(
                    process_path,
                    next_step,
                )

                action_time = 0.0
                if args.mode == "interactive" and automator is not None:
                    if step_index == inconsistent_index:
                        input("[MANUAL] Trigger inconsistent transition, then continue.")
                    else:
                        try:
                            action_time = execute_action(automator, step)
                            input("[MANUAL] Action executed. Continue?")
                        except Exception:
                            input("[MANUAL] Action failed, execute manually then continue.")

                    real_screen: Screen = get_real_screen(automator)

                    if step_index != inconsistent_index:
                        mock_screen = deepcopy(real_screen)
                else:
                    real_screen = get_replay_screen(
                        process_path,
                        step_index,
                        args.replay_real_subdir,
                    )

                visualize, scores, times = get_scores(mock_screen, real_screen, matcher, checker)
                y_true = step_index != inconsistent_index

                save_path = process_visualize_dir / f"{step_index}.jpg"
                cv2.imwrite(str(save_path), visualize)

                y_completed: list[bool] = []
                # Check if we should use VLM retry
                # Either: (1) interactive mode with real agent, or (2) replay mode with skip-agent but has retry data
                use_vlm_retry = False
                step_retry_responses = None
                
                if not y_true:
                    if not args.skip_agent and agent is not None and args.mode == "interactive" and automator is not None:
                        # Case 1: Interactive mode with real agent
                        use_vlm_retry = True
                    elif args.skip_agent and args.mode == "replay":
                        # Case 2: Replay mode with skip-agent, check if retry data exists
                        step_key = f"step_{step_index}"
                        if step_key in vlm_retry_data:
                            step_retry_responses = vlm_retry_data[step_key]
                            use_vlm_retry = step_retry_responses is not None and len(step_retry_responses) > 0
                
                if use_vlm_retry:
                    retries = 3
                    print("  - Deploying action completion agent...")
                    
                    if args.mode == "interactive":
                        automator.back()
                        input("[MANUAL] Confirm backtrack complete, then continue.")
                    
                    # Track which response to use (for replay mode with skip-agent)
                    retry_response_index = 0

                    for attempt in range(retries):
                        y_completed.append(True)
                        
                        # Get real screen
                        if args.mode == "interactive":
                            real_screen = get_real_screen(automator)
                        else:
                            # In replay mode, use the screen from before inconsistency
                            real_screen = get_replay_screen(
                                process_path,
                                step_index,
                                args.replay_real_subdir,
                            )
                        
                        try:
                            if args.skip_agent and step_retry_responses:
                                # Use pre-recorded response instead of calling agent
                                if retry_response_index < len(step_retry_responses):
                                    response = step_retry_responses[retry_response_index]
                                    retry_response_index += 1
                                else:
                                    response = step_retry_responses[-1]  # Use last response if out of range
                                
                                print(f"[VLM (from retry data)]\n{response}")
                                
                                # Process response same way as real agent
                                image = annotate_screen(real_screen)
                                actions, action_names = [], []
                                translator = Translator(real_screen)
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
                                
                                viz_action = (image, response)
                            else:
                                # Use real agent
                                viz_action, action_names, actions = get_action_completion(agent, real_screen, step)
                        except Exception:
                            y_completed[-1] = False
                            continue

                        if actions:
                            action = actions[0]
                            action_name = action_names[0]
                            if action_name != step.action:
                                y_completed[-1] = False

                            try:
                                action_bounds = action()
                            except Exception:
                                y_completed[-1] = False
                                action_bounds = []

                            true_bounds: list[list[int]] = []
                            for _, value in (step.params or {}).items():
                                if isinstance(value, dict):
                                    bounds = value.get("bounds")
                                    if bounds:
                                        true_bounds.append(bounds)

                            if len(true_bounds) != len(action_bounds):
                                y_completed[-1] = False
                            for b1, b2 in zip(true_bounds, action_bounds):
                                if not check_overlap(b1, b2):
                                    y_completed[-1] = False

                            image, response = viz_action
                            inconsistent_dir = process_visualize_dir / "inconsistent"
                            inconsistent_dir.mkdir(parents=True, exist_ok=True)
                            with open(inconsistent_dir / f"{step_index}.txt", "a") as f:
                                f.write(response)
                            image.save(inconsistent_dir / f"{step_index}.jpg")

                        # In replay mode with skip-agent, auto-accept if action is correct
                        if args.skip_agent and args.mode == "replay":
                            if y_completed[-1]:
                                break
                        else:
                            # In interactive mode, ask for manual confirmation
                            verdict = input(
                                f"[MANUAL] Action completion attempt {attempt+1}/{retries}, "
                                f"result = {y_completed[-1]}. Accept? [Y]/[N] "
                            )
                            y_completed[-1] = verdict.strip().lower() == "y"
                            if y_completed[-1]:
                                break

                    if agent is not None and hasattr(agent, 'reset'):
                        agent.reset()

                if not y_true and args.mode == "interactive":
                    break

                score1, score2, score3 = scores
                time1, time2, time3 = times
                writer.writerow([
                    f"{package_name}/{process_name}/{step_index}",
                    f"{score1:.6f}",
                    f"{score2:.6f}",
                    f"{score3:.6f}",
                    f"{action_time:.6f}",
                    f"{time1:.6f}",
                    f"{time2:.6f}",
                    f"{time3:.6f}",
                    y_true,
                    y_completed[-1] if y_completed else False,
                    len(y_completed),
                ])

    print(f"Results written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())