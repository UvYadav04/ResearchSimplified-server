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


class Model:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

        self.model = None
        self.tokenizer = None
        self.client = InferenceClient(
            model=model_name, token=os.environ.get("HF_TOKEN")
        )
        # self.client = OpenAI(
        #     api_key=os.environ.get("GROQ_API_KEY"),
        #     base_url="https://api.groq.com/openai/v1",
        # )

    def initialize_model(self):
        # MODEL_PATH = f"Server/Model/local"
        # if os.path.exists(MODEL_PATH):
        #     model_path = MODEL_PATH
        # else:
        #     model_path = self.model_name
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        # )
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     quantization_config=bnb_config,  # ✅ correct way
        #     dtype=torch.float16,
        # )
        # if not os.path.exists(MODEL_PATH):
        #     self.model.save_pretrained(MODEL_PATH)
        # self.model.eval()
        return self.model

    def get_model(self):
        if self.model is None:
            self.model = self.initialize_model()

        return self.model

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
        # print("message : ", message)
        tokenizer = self.get_tokenizer()
        formatted = self.format_instruction(message)

        return tokenizer.apply_chat_template(
            formatted,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

    async def model_generate(self, message):
        try:
            response = self.client.chat.completions.create(
                messages=self.format_instruction(message),
                stream=False,
                max_tokens=512,
            )

            result = response.choices[0].message.content

            return result

        except Exception as e:
            print("FULL ERROR:", repr(e))
            return None  # ✅ consistent

    def stream_generate(self, inputs, lastContent):
        # model = self.get_model()
        # tokenizer = self.get_tokenizer()

        # streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        # print(inputs)
        # # Run generation in background thread
        # thread = threading.Thread(
        #     target=model.generate,
        #     kwargs={
        #         **inputs,
        #         "streamer": streamer,
        #         "max_new_tokens": 200,
        #         "do_sample": True,
        #         "temperature": 0.7,
        #     },
        # )
        # thread.start()

        type = inputs["type"]

        try:
            if not "content" in inputs:
                return None
            content = inputs["content"]
            stream = self.client.chat.completions.create(
                messages=self.format_instruction(content, lastContent),
                model="openai/gpt-oss-20b",
                stream=True,
                max_tokens=512,
            )
            return stream

        except Exception as e:
            print("FULL ERROR:", repr(e))
            return None

    # elif type == "image":
