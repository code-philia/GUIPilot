import glob
import os
import random
import time
import warnings
from copy import deepcopy
from timeit import default_timer as timer

import cv2
from actions import Automator, Record
from dotenv import load_dotenv
from utils import (
    check_overlap,
    execute_action,
    get_action_completion,
    get_mock_screen,
    get_real_screen,
    get_scores,
)

from guipilot.agent import GPTAgent
from guipilot.checker import GVT as GVTChecker
from guipilot.entities import Screen
from guipilot.matcher import GUIPilotV2 as GUIPilotMatcher

if __name__ == "__main__":
    load_dotenv()
    warnings.filterwarnings("ignore")
    random.seed(42)

    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.getenv("DATASET_PATH")
    process_paths: list[str] = glob.glob(os.path.join(dataset_path, "*", "process_*"))
    process_paths.sort()

    matcher = GUIPilotMatcher()
    checker = GVTChecker()
    automator = Automator()
    agent = GPTAgent(api_key=os.getenv("OPENAI_KEY"))

    # Prepare data recording
    results_dir = os.path.join(base_path, f"results-{int(time.time())}")
    results_txt_path = os.path.join(results_dir, "results.txt")
    os.makedirs(results_dir, exist_ok=False)

    for j, process_path in enumerate(process_paths):
        print(j, process_path)
        record_path = os.path.join(process_path, "record.json")
        record_json: str = open(record_path).read()
        record = Record.model_validate_json(record_json)
        process_no = process_path.split("/")[-1][-1]
        package_name = record.package_name

        try:
            inconsistency_index = random.choice(list(range(0, len(record.steps) - 1)))
            print(f"Inconsistent screen: {inconsistency_index}")
        except IndexError:
            inconsistency_index = -1

        print("Launching app...")
        activity = record.init_activity
        automator.launch(package_name, activity)

        input("[MANUAL] Align phone screen, then continue.")
        for i, step in enumerate(record.steps[:-1]):
            print(
                f"[STEP {i + 1}/{len(record.steps[:-1])}] {record.steps[i].description}"
            )

            # Get the next mock screen
            mock_screen: Screen = get_mock_screen(process_path, record.steps[i + 1])

            # Execution action to transition screen
            action_time = 0
            if i == inconsistency_index:
                input("[MANUAL] Click wrong item to transition to wrong screen")
            else:
                try:
                    action_time = execute_action(automator, step)
                    input("[MANUAL] Executed")
                except:
                    input("[MANUAL] Failed, manual execute")

            # Get real screen after action
            real_screen: Screen = get_real_screen(automator)

            # TEMPORARY
            if i != inconsistency_index:
                mock_screen = deepcopy(real_screen)

            # Check if real screen aligns with mock screen
            visualize, scores, times = get_scores(
                mock_screen, real_screen, matcher, checker
            )
            y_true = i != inconsistency_index

            # Save visualization
            save_path = os.path.join(
                results_dir, f"{package_name}-{process_no}", f"{i}.jpg"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, visualize)

            retries = 3
            y_completed = []
            if y_true == False:
                print("Backtracking...")
                automator.back()
                input("[MANUAL] Check if backtrack correct, then deploy VLM agent")

                for j in range(retries):
                    y_completed.append(True)
                    real_screen: Screen = get_real_screen(automator)
                    try:
                        start_time = timer()
                        visualize, action_names, actions = get_action_completion(
                            agent, real_screen, step
                        )
                        print("completion time: ", timer() - start_time)
                    except:
                        y_completed[-1] = False
                        continue

                    if len(actions) >= 1:
                        action = actions[0]
                        action_name = action_names[0]
                        if action_name != step.action:
                            y_completed[-1] = False

                        try:
                            action_bounds = action()
                        except:
                            y_completed[-1] = False
                            action_bounds = []

                        true_bounds = []
                        for _, value in step.params.items():
                            if isinstance(value, dict):
                                true_bounds.append(value["bounds"])

                        if len(true_bounds) != len(action_bounds):
                            y_completed[-1] = False
                        for b1, b2 in zip(true_bounds, action_bounds):
                            if not check_overlap(b1, b2):
                                y_completed[-1] = False

                        # Save visualization
                        image, response = visualize
                        save_path = os.path.join(
                            results_dir, f"{package_name}-{process_no}"
                        )
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        with open(
                            os.path.join(save_path, f"{i}-inconsistent.txt"), "a"
                        ) as f:
                            f.write(response)
                        image.save(os.path.join(save_path, f"{i}-inconsistent.jpg"))

                    verdict = input(
                        f"[MANUAL] Action completion done, retry = {j + 1}/{retries}, completed = {y_completed[-1]}, verdict [Y]/[N]"
                    )
                    y_completed[-1] = True if verdict.lower() == "y" else False
                    if y_completed[-1]:
                        break

                agent.reset()

            if y_true == False:
                break

            # Record data to csv
            score1, score2, score3 = scores
            time1, time2, time3 = times
            with open(results_txt_path, "a") as f:
                if os.path.getsize(results_txt_path) == 0:
                    f.write(
                        ",".join(
                            [
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
                            ]
                        )
                    )

                f.write(
                    ",".join(
                        [
                            f"{package_name}/{process_no}/{i}",
                            str(score1),
                            str(score2),
                            str(score2),
                            str(action_time),
                            str(time1),
                            str(time2),
                            str(time3),
                            str(y_true),
                            str(y_completed[-1] if len(y_completed) > 0 else False),
                            str(len(y_completed)),
                        ]
                    )
                )
                f.write("\n")
