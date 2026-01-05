import base64
import io
import os
from abc import ABC, abstractmethod
from typing import Optional

import openai
from PIL import Image


class Agent(ABC):
    @abstractmethod
    def __call__(self, prompt: str, images: Optional[list[Image.Image]] = None) -> str:
        pass


class GPTAgent(Agent):
    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        base_path = os.path.abspath(os.path.dirname(__file__))
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        self.system_prompt = open(f"{base_path}/action_completion.system.prompt").read()
        self.history = [{"role": "system", "content": self.system_prompt}]

    def reset(self):
        self.history = []

    def __call__(self, prompt: str, images: Optional[list[Image.Image]] = None) -> str:
        if len(self.history) > 0:
            self.history.append(
                {
                    "role": "user",
                    "content": "Incorrect, are you sure it is the correct: 1) widget id, 2) action type 3) direction (i.e., swipe/scroll)? Do not use the same answer again.",
                }
            )

        user_prompt = [{"type": "text", "text": prompt}]

        for image in images:
            bytes = io.BytesIO()
            image.save(bytes, format="JPEG")
            image_b64 = base64.b64encode(bytes.getvalue()).decode("ascii")
            user_prompt.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )

        self.history.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            max_tokens=1024,
            temperature=0,
            top_p=1,
        )

        content = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": content})

        return content
