import json
import os

import cv2
from dotenv import load_dotenv
from utils import (
    check_action,
    get_action_completion,
    get_report,
    get_scores,
    get_screen,
    visualize,
)

from guipilot.agent import GPTAgent
from guipilot.checker import GVT as GVTChecker
from guipilot.entities import Screen
from guipilot.matcher import GUIPilotV2 as GUIPilotMatcher


def get_processes(mockups_path: str) -> list[str]:
    """Get all paths of the nested processes in the case study dataset"""
    parent_dirs = set()
    for root, dirs, _ in os.walk(mockups_path):
        if "mockup" in dirs:
            full_path = os.path.abspath(root)
            parent_dirs.add(full_path)

    return sorted(parent_dirs)


base_path = os.path.dirname(os.path.abspath(__file__))
mockups_path = os.getenv("DATASET_PATH")
process_paths = get_processes(mockups_path)
load_dotenv()

for p, process_path in enumerate(process_paths):
    process_id = process_path.replace(mockups_path, "")
    print(f"{process_id}")

    agent = GPTAgent(api_key=os.getenv("OPENAI_KEY"))
    matcher = GUIPilotMatcher()
    checker = GVTChecker()

    implementation_path = os.path.join(process_path, "implementation")
    mockup_path = os.path.join(process_path, "mockup")
    json_path = os.path.join(implementation_path, "process.json")
    process: list = json.load(open(json_path, "r"))

    for s, step in enumerate(process):
        if s != 4:
            continue
        screen_filename: str = step["screen"]
        if "branch" in screen_filename:
            continue
        real_screen: Screen = get_screen(implementation_path, screen_filename)
        mock_screen: Screen = get_screen(
            mockup_path, screen_filename.replace(".jpg", ".png")
        )
        print(f"[>] Screen {p}-{s + 1}")

        # Match widgets and check for inconsistencies
        pairs, score, match_time = get_scores(mock_screen, real_screen, matcher)
        inconsistencies, check_time = checker.check(mock_screen, real_screen, pairs)
        image, bbox_image, match_image = visualize(
            mock_screen, real_screen, pairs, inconsistencies
        )
        print(f"\t[>] Score: {score}, {len(inconsistencies)} inconsistencies")

        # Use VLM agent to get actions that lead to next screen
        mock_actions = step["mock_actions"]
        true_actions = step["actions"]
        action_correct, action_trials = False, []
        print(f"\t[>] Mock actions: {mock_actions}")

        while not action_correct and len(action_trials) < 3:
            agent_image, action_names, actions_raw, actions = get_action_completion(
                agent, real_screen, mock_actions
            )
            print(f"\t\t[>] VLM agent trial {len(action_trials) + 1}/3: ", actions_raw)

            if len(actions) != len(true_actions):
                print("\t\t[>]Incorrect length")
                action_trials.append(actions_raw)
                continue

            if not all(
                check_action(true_action, action_name, action)
                for true_action, action_name, action in zip(
                    true_actions, action_names, actions
                )
            ):
                action_trials.append(actions_raw)
                continue

            action_trials.append(actions_raw)
            action_correct = True

        # Visualize results
        output_path = os.path.join(base_path, "output", f"process{p}", f"screen{s}")
        os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, "image.jpg"), image)
        cv2.imwrite(os.path.join(output_path, "image_bbox.jpg"), bbox_image)
        cv2.imwrite(os.path.join(output_path, "image_match.jpg"), match_image)
        cv2.imwrite(os.path.join(output_path, "image_agent.jpg"), agent_image)

        # Generate report
        report_path = os.path.join(output_path, "report.json")
        report = get_report(
            process_path,
            s,
            match_time,
            check_time,
            pairs,
            inconsistencies,
            action_correct,
            action_trials,
        )
        with open(report_path, "w") as f:
            f.write(report)
