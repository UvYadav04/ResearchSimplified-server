import google.generativeai as genai
import os
from huggingface_hub import InferenceClient
from openai import OpenAI

async def awake_gemini():
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    return genai.GenerativeModel("gemini-2.5-flash")

async def awake_groq():
    return  OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

async def awake_client():
    return InferenceClient(
        oken=os.environ.get("HF_TOKEN")
    )
