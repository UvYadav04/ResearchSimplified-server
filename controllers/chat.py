import os
import requests
from startupFunctions import get_gemini, get_client
from fastapi import Request
from Model.model import Model
from fastapi.responses import StreamingResponse
from SafeExecution.safeExecution import safeExecution
import asyncio
import json


@safeExecution
async def classifyQuery(query: str):
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    }
    payload = {
        "inputs": query,
        "parameters": {
            "candidate_labels": [
                "A standalone general query",
                "A request to summarize content",
                "A follow-up query depending on previous context",
                "A request to generate an image",
            ]
        },
    }
    response = requests.post(
        os.environ["CLASSIFICATION_API_URL"], headers=headers, json=payload
    )
    return response.json()

@safeExecution
async def handleQuery(query: str, relatedContext: str, request: Request):
    # gemini = get_gemini(request.app)
    print("in conroller")
    client = await get_client(request.app)
    modelManager = Model(client)

    streamer = modelManager.stream_query(query, relatedContext)

    async def streamer_generator():
        for chunk in streamer:
            choices = chunk.choices
            if choices is None or len(choices)==0:
                continue
            text =  choices[0].delta.content
            if text:
                yield json.dumps({"type": "text", "content": text}) + "<END>"
                await asyncio.sleep(0)

    return streamer_generator
