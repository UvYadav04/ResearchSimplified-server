from google import genai
import os
from huggingface_hub import InferenceClient
from openai import OpenAI


def awake_gemini():
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return client


async def awake_groq():
    return  OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

async def awake_client():
    return InferenceClient(
        token=os.environ.get("HF_TOKEN")
    )
