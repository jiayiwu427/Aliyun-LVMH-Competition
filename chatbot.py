from typing import List
import os
from openai import OpenAI  # Assuming you have a client from the Dashscope library

class QWENChatBot:
    def __init__(self, system_message: str):
        # Initialize the Dashscope client
        self.client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.messages = self.init_messages(system_message)

    def init_messages(self, system_message: str) -> List[dict]:
        # Initialize with system message
        messages = [{'role': 'system', 'content': system_message}]
        return messages

    def ask(self, message: str, temp=0, top_p=1) -> str:
        # Manage message history
        if len(self.messages) >= 7:
            self.messages.pop(1)  # Remove old messages if limit exceeded
            self.messages.pop(1)

        # Append new user message
        self.messages.append({"role": "user", "content": message})

        # Make API call to Dashscope LLM (qwen-plus model)
        rsp = self.client.chat.completions.create(
            model="qwen-plus",  # Change model to qwen-plus or other available model
            messages=self.messages,
            temperature=temp,
            top_p=top_p,
            n=1
        )

        # Extract response and append it to the conversation history
        rsp = rsp.choices[0].message.content
        self.messages.append({"role": "assistant", "content": rsp})

        return rsp