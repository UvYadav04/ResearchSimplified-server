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
from utils.idGenerator import generateId


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
async def handleQuery(query: str, request: Request, chunkId: str | None):
    # gemini = get_gemini(request.app)
    session_id = getattr(request.state, "session_id", None)
    user_id = getattr(request.state, "user_id", None)
    print("sessionId", session_id)
    print("userId", user_id)
    if user_id is None or session_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized call to chat"
        )

    print("in conroller")
    client = await get_client(request.app)
    redis_client = get_redis(request.app)
    embedder = get_fastembed(request.app)
    redis = Redis(redis=redis_client, session_id=session_id)
    modelManager = Model(client)

    embedding = list(embedder.embed([query]))[0]
    # print(embedding)
    relatedContext = redis.search(embedding, top_k=5, doc_type="chunk")
    relatedChat = redis.search(embedding, top_k=5, doc_type="message")
    print("related Chat : ",relatedChat)
    contextChunk = None
    if chunkId is not None:
        contextChunk = redis.get_chunk_by_id(chunkId)

    streamer = modelManager.stream_query(
        query, relatedContext, relatedChat, contextChunk
    )

    async def streamer_generator():
        chatResponse = ""
        for chunk in streamer:
            choices = chunk.choices
            if choices is None or len(choices) == 0:
                continue
            text = choices[0].delta.content
            if text:
                chatResponse += text
                yield json.dumps({"type": "text", "content": text}) + "<END>"
                await asyncio.sleep(0)
        embeddings = list(embedder.embed([query, chatResponse]))
        redis.add_to_chat(
            messages=[
                {"message_id": generateId(), "text": query},
                {"message_id": generateId(), "text": chatResponse},
            ],
            embeddings=embeddings,
        )

    return streamer_generator
