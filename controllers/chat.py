import os
import requests
from startupFunctions import get_gemini, get_client, get_redis, get_fastembed
from fastapi import Request
from fastapi import HTTPException, status
from Model.model import Model
from fastapi.responses import StreamingResponse
from SafeExecution.safeExecution import safeExecution
import asyncio
import json
from Redis.redis import Redis


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
async def handleQuery(query: str, request: Request):
    # gemini = get_gemini(request.app)
    session_id = getattr(request.state, "session_id", None)
    user_id = getattr(request.state, "user_id", None)
    print("sessionId",session_id)
    print("userId",user_id)
    if user_id is None or session_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized call to chat"
        )

    print("in conroller")
    client = await get_client(request.app)
    redis_client = get_redis(request.app)
    embedder = get_fastembed(request.app)
    redis = Redis(redis=redis_client,session_id=session_id)
    modelManager = Model(client)

    embedding = list(embedder.embed([query]))[0]
    # print(embedding)
    relatedContext = redis.search(embedding,top_k=5,doc_type="chunk")

    streamer = modelManager.stream_query(query, relatedContext)

    async def streamer_generator():
        for chunk in streamer:
            choices = chunk.choices
            if choices is None or len(choices) == 0:
                continue
            text = choices[0].delta.content
            if text:
                yield json.dumps({"type": "text", "content": text}) + "<END>"
                await asyncio.sleep(0)

    return streamer_generator
