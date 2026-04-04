import os
import requests
from startupFunctions import get_gemini
from fastapi import Request
from Model.model import Model
from fastapi.responses import StreamingResponse

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



async def handleQuery(query:str,relatedContext:str,request:Request):
    gemini = await get_gemini(request.app)
    modelManager = Model(gemini)
    streamer = modelManager.stream_query(query,relatedContext)

    def streamer_generator():
        for token in streamer:
            if token:
                yield token

    return StreamingResponse(streamer_generator(),media_type='text/plain')