from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading


class Model:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

        self.model = None
        self.tokenizer = None

    def get_model(self):
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)

        return self.model

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return self.tokenizer

    def format_instruction(self, message: str):
        return [
            {
                "role": "system",
                "content": (
                    "You are a Research paper content simplifier. "
                    "Simplify the given content. If already simple, return as is."
                ),
            },
            {"role": "user", "content": message},
        ]

    def apply_chat_template(self, message: str):
        tokenizer = self.get_tokenizer()

        formatted = self.format_instruction(message)

        return tokenizer.apply_chat_template(
            formatted,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

    def stream_generate(self, message: str):
        model = self.get_model()
        tokenizer = self.get_tokenizer()

        inputs = tokenizer.apply_chat_template(
            self.format_instruction(message),
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

        # Run generation in background thread
        thread = threading.Thread(
            target=model.generate,
            kwargs={
                "inputs": inputs,
                "streamer": streamer,
                "max_new_tokens": 200,
                "do_sample": True,
                "temperature": 0.7,
            },
        )
        thread.start()

        # Yield tokens as they arrive
        for token in streamer:
            yield token
