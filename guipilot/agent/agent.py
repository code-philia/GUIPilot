import base64
import io
import os
from abc import ABC, abstractmethod
from typing import Optional

import openai
from PIL import Image
import requests  # 新增导入
import json  # 新增导入


class Agent(ABC):
    @abstractmethod
    def __call__(self, prompt: str, images: Optional[list[Image.Image]] = None) -> str:
        pass


class QwenAgent(Agent):
    def __init__(self, api_key: str, model: str = "qwen-vl-plus") -> None:
        base_path = os.path.abspath(os.path.dirname(__file__))
        self.model = model
        self.api_key = api_key
        
        # 使用正确的API端点
        # 方案1：使用兼容OpenAI的端点
        self.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        
        # 方案2：或者使用多模态生成端点
        # self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        
        self.system_prompt = open(f"{base_path}/action_completion.system.prompt").read()
        self.history = [{"role": "system", "content": self.system_prompt}]
        
        print(f"[INFO] 使用Qwen API: {self.api_url}")

    def __call__(self, prompt: str, images: Optional[list[Image.Image]] = None) -> str:
        if len(self.history) > 0:
            self.history.append({
                "role": "user", 
                "content": "Incorrect, are you sure it is the correct: 1) widget id, 2) action type 3) direction (i.e., swipe/scroll)? Do not use the same answer again."
            })

        # 准备消息
        messages = self.history.copy()
        
        # 构建当前用户消息
        user_content = []
        
        # 添加文本
        user_content.append({
            "type": "text",
            "text": prompt
        })
        
        # 添加图片
        if images:
            for image in images:
                bytes_io = io.BytesIO()
                image.save(bytes_io, format="JPEG")
                image_b64 = base64.b64encode(bytes_io.getvalue()).decode('ascii')
                
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                })
        
        messages.append({
            "role": "user",
            "content": user_content
        })

        # 准备请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 尝试两种不同的请求格式
        try:
            # 格式1：兼容OpenAI格式
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0,
                "top_p": 1
            }
            
            print(f"[DEBUG] 发送请求到: {self.api_url}")
            print(f"[DEBUG] 模型: {self.model}")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            print(f"[DEBUG] 响应状态码: {response.status_code}")
            print(f"[DEBUG] 响应内容: {response.text[:500]}...")
            
            response.raise_for_status()
            result = response.json()
            
            # 提取响应内容
            if "choices" in result:
                content = result["choices"][0]["message"]["content"]
            elif "output" in result and "choices" in result["output"]:
                content = result["output"]["choices"][0]["message"]["content"]
            else:
                content = str(result)
                
            # 更新历史
            self.history.append({"role": "user", "content": user_content})
            self.history.append({"role": "assistant", "content": content})
            
            return content
            
        except Exception as e:
            print(f"[ERROR] Qwen API调用失败: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"[ERROR] 响应内容: {e.response.text}")
            
            # 如果兼容模式失败，尝试原始格式
            try:
                print("[INFO] 尝试使用原始Qwen格式...")
                return self._call_original_format(prompt, images)
            except Exception as e2:
                print(f"[ERROR] 原始格式也失败: {e2}")
                raise

    def _call_original_format(self, prompt: str, images: Optional[list[Image.Image]] = None) -> str:
        """使用Qwen原始API格式"""
        original_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        
        # 构建输入
        input_data = {
            "messages": []
        }
        
        # 构建消息
        message_content = []
        message_content.append({"text": prompt})
        
        if images:
            for image in images:
                bytes_io = io.BytesIO()
                image.save(bytes_io, format="JPEG")
                image_b64 = base64.b64encode(bytes_io.getvalue()).decode('ascii')
                
                message_content.append({
                    "image": f"data:image/jpeg;base64,{image_b64}"
                })
        
        input_data["messages"].append({
            "role": "user",
            "content": message_content
        })
        
        payload = {
            "model": self.model,
            "input": input_data,
            "parameters": {
                "max_tokens": 1024,
                "temperature": 0,
                "top_p": 1
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            original_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        if "output" in result and "choices" in result["output"]:
            content = result["output"]["choices"][0]["message"]["content"]
        else:
            content = str(result)
        
        return content


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
