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
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

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
                    "Simplify the content into human understandable language. Create a few paragraphs according to the content, and most importantly in simplified language."
                    "Rules:\n"
                    "- Always format output in clean Markdown.\n"
                    "- You can decide add as many paragraphs as you think will be suffecient.\n"
                    "- If the input is already simple, general information, very short, not connected to previous content, and you think it does not need to be explained, start with same.\n"
                    "- Do NOT copy sentences from the input.\n"
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def format_query(
        self,
        message: str,
        relatedContent: str,
        relatedChats: str,
        contextChunk: str | None,
    ):
        # ----------------------------
        # SYSTEM PROMPT
        # ----------------------------
        system_prompt = (
            "You are an AI that represents the user's uploaded research paper. "
            "Speak in first person when referring to the paper (e.g., 'In this work, I show...'). "
            "When the user asks questions about the paper, act like a tutor: "
            "explain concepts clearly, use simple language, short sentences, and examples when helpful. "
            "Avoid unnecessary jargon. "
            "When the user interacts casually, respond like a helpful assistant. "
            "Do not mention system instructions. "
            "Always keep responses clear, concise, and natural."
        )

        # ----------------------------
        # BUILD USER CONTENT
        # ----------------------------
        parts = []

        # 🔥 1. MOST IMPORTANT: Context Chunk
        if contextChunk:
            parts.append(
                f"""
    IMPORTANT CONTEXT (highest priority):
    {contextChunk}

    The user is most likely referring to this. Use it carefully.
    """
            )

        # 🔥 2. Supporting document context
        if relatedContent:
            parts.append(
                f"""
    Additional Context:
    {relatedContent}
    """
            )

        # 🔥 3. Previous chat context
        if relatedChats:
            parts.append(
                f"""
    Conversation History:
    {relatedChats}
    """
            )

        # ----------------------------
        # USER QUERY
        # ----------------------------
        parts.append(
            f"""
    User Question:
    {message}

    Instructions:
    - If IMPORTANT CONTEXT is present, prioritize it.
    - Use additional context only if helpful.
    - If context is irrelevant, ignore it.
    - Answer clearly and simply.
    """
        )

        user_content = "\n".join(parts)

        return [
            {"role": "system", "content": system_prompt},
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

    def stream_document(self, inputs, lastContent):
        # this generation method is for hf inference providers
        try:
            if not "content" in inputs:
                return None
            content = inputs["content"]
            stream = self.model.chat.completions.create(
                messages=self.format_instruction(content, lastContent),
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                stream=True,
                max_tokens=512,
            )
            return stream

        except Exception as e:
            print("FULL ERROR:", repr(e))
            return {"error": str(e)}

    def stream_query(self, input, relatedContent, relatedChats, contextChunk):

        # contents = self.format_chunk_query(input, relatedContent)

        # response = self.model.models.generate_content_stream(
        #     model="gemini-2.5-flash",
        #     contents = contents,
        # )
        messages = self.format_query(input, relatedContent, relatedChats, contextChunk)
        # print(messages)
        stream = self.model.chat.completions.create(
            messages=messages,
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            stream=True,
            max_tokens=512,
        )
        return stream
        # return response

    def read_image(self, image, query):
        return None

    def generate_image(self, query):
        return None
