import time
from enum import Enum
from typing import Literal

import uiautomator2 as u2
from uiautomator2 import Device


class Action(Enum):
    BACK = "back"
    CLICK = "click"
    LONG_CLICK = "long_click"
    SEND_KEYS = "send_keys"
    SCROLL = "scroll"
    SWIPE = "swipe"
    DRAG = "drag"


class Automator:
    def __init__(
        self,
        device: Device = u2.connect("192.168.240.112:5555"),
        wait_until_loaded: bool = False,
    ) -> None:
        """Automator wraps a uiautomator device and
        defines a fixed set of actions on the connected phone.
        """
        self.wait_until_loaded = wait_until_loaded
        self.device = device

    def wait(self, timeout=30, interval=1, stability_threshold=5):
        """Wait for screen to finish loading"""
        if not self.wait_until_loaded:
            return
        start_time = time.time()
        previous_snapshot = None
        stability = 0

        while time.time() - start_time < timeout:
            current_snapshot = self.device.dump_hierarchy()
            stability += 1 if current_snapshot == previous_snapshot else 0
            if stability >= stability_threshold:
                return
            previous_snapshot = current_snapshot
            time.sleep(interval)

    def launch(self, package_name: str, activity: str):
        """Launch an app and move to activity"""
        self.device.app_start(package_name, activity, stop=True, wait=True)

    def back(self):
        """Press on the 'back' hard/soft key"""
        self.device.press("back")
        self.wait()

    def click(self, locator: dict):
        """Click and release."""
        if "x" in locator and "y" in locator:
            x = int(locator["x"])
            y = int(locator["y"])
            self.device.click(x, y)
            return

        target = self.device(**locator)
        target.click()
        self.wait()

    def long_click(self, locator: dict):
        """Click and hold for a duration."""
        target = self.device(**locator)
        target.long_click()
        self.wait()

    def send_keys(self, text: str):
        """Simulate typing on-screen keyboard."""
        self.device.send_keys(text=text, clear=True)
        self.wait()

    def scroll(
        self, direction: Literal["left", "right", "up", "down"], distance: int = 2
    ):
        """Scroll until the end or the content is fully replaced with new content (i.e., 1 screen length)."""
        params = {
            "down": [0.5, 0.7, 0.5, 0.2],
            "up": [0.5, 0.2, 0.5, 0.7],
            "left": [0.2, 0.5, 0.7, 0.5],
            "right": [0.7, 0.5, 0.2, 0.5],
        }[direction]

        for _ in range(distance):
            self.device.swipe(*params)
        self.wait()

    def swipe(self, selector: dict, direction: Literal["left", "right", "up", "down"]):
        """Swipe from the center of the widget to its edge."""
        target = self.device(**selector)
        target.swipe(direction)
        self.wait()

    def drag(self, selector1: dict, selector2: dict) -> bool:
        """Drag widget to the center of another widget."""
        from_widget = self.device(**selector1)
        from_widget.drag_to(**selector2)
        self.wait()
