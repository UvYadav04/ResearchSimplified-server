from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from huggingface_hub import InferenceClient

import torch
import threading
from openai import OpenAI
import os
import google.generativeai as genai
from PIL import Image
import io


class Model:
    def __init__(self, model, client, device: str = "cpu"):
        self.model = model
        self.device = device
        self.tokenizer = None

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def format_instruction(self, message: str, lastContent: str = ""):
        user_content = message

        if lastContent != "":
            user_content = f"""
                    Previous context:
                    {lastContent}

                    Current content:
                    {message}

                    Instructions:
                    - If relevant, connect the current content with the previous context.
                    - Otherwise, ignore previous context.
            """

        return [
            {
                "role": "system",
                "content": (
                    "You are an expert at simplifying complex research content for beginners. "
                    "Your goal is to make the explanation easy to understand for a normal user "
                    "with no technical background.\n\n"
                    "Follow this structure strictly:\n\n"
                    "1. Simplified Explanation:\n"
                    "- Explain the idea in very simple language.\n"
                    "- Use analogies or real-life examples when possible.\n"
                    "- Avoid jargon. If needed, explain it in simple words.\n\n"
                    "2. Key Points:\n"
                    "- Provide 3–6 bullet points.\n"
                    "- Keep them short and clear.\n\n"
                    "3. Why It Matters:\n"
                    "- Briefly explain why this concept is useful or important in real life.\n\n"
                    "Rules:\n"
                    "- Do NOT copy sentences from the input.\n"
                    "- Do NOT use complex words unnecessarily.\n"
                    "- Keep it concise but clear.\n"
                    "- If the input is already simple or irrelevant, return the same input.\n"
                    "- Always format output in clean Markdown.\n"
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def apply_template(self, message: str):
        tokenizer = self.get_tokenizer()
        formatted = self.format_instruction(message)

        return tokenizer.apply_chat_template(
            formatted,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

    def stream_generate(self, inputs, lastContent):
        # this generation method is for hf inference providers
        try:
            if not "content" in inputs:
                return None
            content = inputs["content"]
            stream = self.model.chat.completions.create(
                messages=self.format_instruction(content, lastContent),
                model="openai/gpt-oss-20b",
                stream=True,
                max_tokens=512,
            )
            return stream

        except Exception as e:
            print("FULL ERROR:", repr(e))
            return None

    def read_image(self, image, query):
        return None

    def generate_image(self, query):
        return None
