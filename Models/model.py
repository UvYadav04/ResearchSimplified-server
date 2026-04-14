from google import genai
import os
from huggingface_hub import InferenceClient
import redis
from openai import OpenAI
from Redis.schema import create_index
from fastembed import TextEmbedding
import cohere


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

def awake_redis():
    redis_client =  redis.Redis(host="localhost", port=6379, decode_responses=False)
    create_index(redis_client=redis_client)
    return redis_client

def awake_fastembed():
     return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def awake_coherent():
    return cohere.ClientV2(
        api_key=os.environ["COHERE_API"]
    )

def awake_falAi():
    return InferenceClient(
        provider="fal-ai",
        api_key=os.environ["HF_TOKEN"],
    )
