import copy
import glob
import inspect
import json
import os
from timeit import default_timer as timer
from typing import Callable, Literal, Optional

import uiautomator2 as u2
from pydantic import BaseModel, model_validator
from typing_extensions import Self
from uiautomator2 import Device, UiObject

from .automator import Action, Automator


class Step(BaseModel):
    activity: str
    screenshot: str
    description: str
    layout: str
    action: Optional[str] = None
    params: Optional[dict] = None
    time: Optional[float] = None


class Record(BaseModel):
    package_name: str
    package_version: str
    init_activity: str
    steps: list[Step]

    @model_validator(mode="after")
    def validate(self) -> Self:
        if len(self.steps) <= 0:
            raise ValueError("Record must have >0 steps")

        if self.init_activity != self.steps[0].activity:
            raise ValueError("Initial activity must match activity in the first step")

        for i, step in enumerate(self.steps):
            if step.screenshot != f"{i+1}.jpg":
                raise ValueError("Screenshot filenames must be a sequential number JPG")

            if step.layout != f"{i+1}.xml":
                raise ValueError("Layout filenames must be a sequential number XML")

            if step.action is None and i != len(self.steps) - 1:
                raise ValueError("An empty action can only be at the last step")

            if (
                step.action is not None
                and step.action.upper() not in Action._member_names_
            ):
                raise ValueError(
                    f"Action must be one of the atomic actions, got {step.action}"
                )

            if step.time is not None and step.time < 0:
                raise ValueError("Time taken must be a non-negative float")

            if step.params is None:
                continue
            for _, value in step.params.items():
                if isinstance(value, dict) and "x" not in value and "y" not in value:
                    if value.get("bounds") is None:
                        raise ValueError("Selector must have bounds information")

        return self


class Recorder(Automator):
    def __init__(self, device: Device = u2.connect("192.168.240.112:5555")) -> None:
        """Recorder extends the Automator and
        records all actions, screenshots, layouts in a record file.
        Meant to be used in CLI for manually recording steps.
        """
        super().__init__(device)

    def find(self, **kwargs) -> None:
        """List all uiobjects matching criteria"""
        uiobjects: list[UiObject] = self.device(**kwargs)
        for i, uiobject in enumerate(uiobjects):
            info: dict = uiobject.info
            bbox = uiobject.info["visibleBounds"]
            x = bbox["left"]
            y = bbox["top"]
            width = bbox["right"] - bbox["left"]
            height = bbox["bottom"] - bbox["top"]
            print(
                i,
                [x, y, width, height],
                info["className"],
                info["contentDescription"],
                info["text"],
                "\n",
            )

    def reset(self, package_name: str = None) -> None:
        """Reset app and auto grant all permissions"""
        if package_name is None:
            app_info = self.device.app_current()
            package_name = app_info.get("package", None)
        self.device.app_clear(package_name)
        self.device.app_auto_grant_permissions(package_name)

    def start(self):
        """Start recording, get current app data"""
        # Get app information
        app_info = self.device.app_current()
        self.package_name = app_info.get("package", None)
        self.package_version = (
            self.device.shell(f"dumpsys package {self.package_name} | grep versionName")
            .output.strip()
            .replace("versionName=", "")
        )
        self.init_activity = app_info.get("activity", None)

        # Create directory to store traces
        base_path = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(base_path, self.package_name)
        os.makedirs(app_path, exist_ok=True)
        process_no = (
            sum(
                os.path.isdir(os.path.join(app_path, entry))
                for entry in os.listdir(app_path)
            )
            + 1
        )
        self.record_dir = os.path.join(app_path, f"process_{process_no}")
        os.makedirs(self.record_dir, exist_ok=False)

        record = {}
        record["package_name"] = self.package_name
        record["package_version"] = self.package_version
        record["init_activity"] = self.init_activity
        record_path = os.path.join(self.record_dir, "record.json")
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)

    def stop(self):
        """Stop recording, get final screen, layout, activity"""
        layout = self.device.dump_hierarchy()
        screenshot = self.device.screenshot()
        activity = self.device.app_current().get("activity", None)
        step_no = len(glob.glob(f"{self.record_dir}/*.jpg")) + 1
        record_path = os.path.join(self.record_dir, "record.json")
        layout_path = os.path.join(self.record_dir, f"{step_no}.xml")
        screenshot_path = os.path.join(self.record_dir, f"{step_no}.jpg")
        with open(layout_path, "w") as f:
            f.write(layout)
        screenshot.save(screenshot_path)
        step = Step(
            activity=activity,
            description="",
            screenshot=screenshot_path.split("/")[-1],
            layout=layout_path.split("/")[-1],
            action=None,
            params=None,
            time=None,
        )
        record: dict = json.load(open(record_path, "r"))
        record["steps"] = record.get("steps", []) + [step.__dict__]
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)

    def undo(self):
        """Undo last recorded step"""
        step_no = len(glob.glob(f"{self.record_dir}/*.jpg"))
        if step_no <= 0:
            return
        record_path = os.path.join(self.record_dir, "record.json")
        layout_path = os.path.join(self.record_dir, f"{step_no}.xml")
        screenshot_path = os.path.join(self.record_dir, f"{step_no}.jpg")
        os.remove(layout_path)
        os.remove(screenshot_path)
        record: dict = json.load(open(record_path, "r"))
        steps: list = record["steps"]
        steps.pop()
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2)

    def record(action: Callable):
        def wrapper(self: "Recorder", *args, **kwargs):
            # track all parameters, add bbox info to selectors
            params = {}
            sig = inspect.signature(action)
            bound_arguments = sig.bind(self, *args, **kwargs)
            bound_arguments.apply_defaults()
            arguments_dict = bound_arguments.arguments

            for name, value in arguments_dict.items():
                if name == "self":
                    continue

                _value = copy.deepcopy(value)

                if isinstance(_value, dict) and "x" not in _value and "y" not in _value:
                    uiobject = self.device(**_value)
                    if len(uiobject) == 1:
                        _value["bounds"] = [
                            int(uiobject.info["visibleBounds"][key])
                            for key in ["left", "top", "right", "bottom"]
                        ]
                    else:
                        raise ValueError(
                            f"Error: selector {name} must have 1 target, found {len(uiobject)}"
                        )

                params[name] = _value

            # before executing action, record data
            start_time = timer()
            layout = self.device.dump_hierarchy()
            screenshot = self.device.screenshot()
            activity = self.device.app_current().get("activity", None)

            # action
            action(self, *args, **kwargs)

            # after executing action, save step
            time = timer() - start_time
            step_no = len(glob.glob(f"{self.record_dir}/*.jpg")) + 1
            record_path = os.path.join(self.record_dir, "record.json")
            layout_path = os.path.join(self.record_dir, f"{step_no}.xml")
            screenshot_path = os.path.join(self.record_dir, f"{step_no}.jpg")

            with open(layout_path, "w") as f:
                f.write(layout)
            screenshot.save(screenshot_path)
            step = Step(
                activity=activity,
                screenshot=screenshot_path.split("/")[-1],
                description="",
                layout=layout_path.split("/")[-1],
                action=action.__name__,
                params=params,
                time=time,
            )
            record: dict = json.load(open(record_path, "r"))
            record["steps"] = record.get("steps", []) + [step.__dict__]
            with open(record_path, "w") as f:
                json.dump(record, f, indent=2)

        return wrapper

    @record
    def back(self):
        super().back()

    @record
    def click(self, locator: dict):
        super().click(locator)

    @record
    def long_click(self, locator: dict):
        super().long_click(locator)

    @record
    def send_keys(self, text: str):
        super().send_keys(text)

    @record
    def scroll(
        self, direction: Literal["left", "right", "up", "down"], distance: int = 2
    ):
        super().scroll(direction, distance)

    @record
    def swipe(self, selector: dict, direction: Literal["left", "right", "up", "down"]):
        super().swipe(selector, direction)

    @record
    def drag(self, selector1: dict, selector2: dict) -> bool:
        super().drag(selector1, selector2)
