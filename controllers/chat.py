import os
import requests
from startupFunctions import (
    get_gemini,
    get_client,
    get_redis,
    get_fastembed,
    get_coherent,
    get_falAI,
    get_groq,
)
from fastapi import Request
from fastapi import HTTPException, status
from Model.model import Model
from Model.falAi import FalAi
from fastapi.responses import StreamingResponse
from SafeExecution.safeExecution import safeExecution
import asyncio
import json
from Redis.redis import Redis
from utils.idGenerator import generateId
from PIL import Image


@safeExecution
async def classifyQuery(query: str,options):
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    }
    payload = {
        "inputs": query,
        "parameters": {
            "candidate_labels": options
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
    if user_id is None or session_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized call to chat"
        )

    client = await get_client(request.app)
    redis_client = get_redis(request.app)
    embedder = get_fastembed(request.app)
    cohere = get_coherent(request.app)
    redis = Redis(redis=redis_client, cohere_client=cohere, session_id=session_id)
    modelManager = Model(client)

    embedding = list(embedder.embed([query]))[0]
    relatedContext = redis.search(query, embedding, top_k=20, doc_type="chunk")
    relatedChat = redis.search(query, embedding, top_k=20, doc_type="message")
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


@safeExecution
async def generateImage(query: str, request: Request, chunkId: str | None):
    session_id = getattr(request.state, "session_id", None)
    user_id = getattr(request.state, "user_id", None)
    if user_id is None or session_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized call to chat"
        )
    if chunkId is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please select a valid context",
        )
    redis_client = get_redis(request.app)
    cohere = get_coherent(request.app)
    redis = Redis(redis=redis_client, cohere_client=cohere, session_id=session_id)
    contextChunk = redis.get_chunk_by_id(chunkId)
    groq = await get_groq(request.app)
    model = Model(groq)
    prompt = model.generate_diffusion_prompt(query, contextChunk)

    fal_ai_client = get_falAI(request.app)
    falAi = FalAi(fal_ai_client)
    image = falAi.generate_image(prompt)
    Image.open(image)
