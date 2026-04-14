import os
import requests
import logging
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
from Model.llm import LLM


# Configure logger
logger = logging.getLogger("chatController")
logging.basicConfig(level=logging.INFO)


@safeExecution
async def classifyQuery(query: str, options):
    try:
        logger.info("classifyQuery called.")
        headers = {
            "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
        }

        payload = {
            "inputs": query,
            "parameters": {"candidate_labels": options},
        }
        logger.debug(
            f"Sending classification request to {os.environ['CLASSIFICATION_API_URL']}."
        )

        response = requests.post(
            os.environ["CLASSIFICATION_API_URL"],
            headers=headers,
            json=payload,
            timeout=10,
        )

        # 🔥 Check status FIRST
        if response.status_code != 200:
            logger.error(f"HF API error: {response.status_code}, {response.text}")
            return {"error": "HF API failed", "raw": response.text}

        # 🔥 Check empty response
        if not response.text.strip():
            logger.error("Empty response from HF")
            return {"error": "Empty response"}

        logger.info("classifyQuery successful and returning result.")
        return response.json()

    except Exception as e:
        logger.error(f"classifyQuery error: {str(e)}")
        return {"error": str(e)}


@safeExecution
async def handleQuery(query: str, request: Request, chunkId: str | None):
    logger.info("handleQuery endpoint called.")
    # gemini = get_gemini(request.app)
    session_id = getattr(request.state, "session_id", None)
    user_id = getattr(request.state, "user_id", None)
    if user_id is None or session_id is None:
        logger.warning("Unauthorized call to chat")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized call to chat"
        )

    redis_client = get_redis(request.app)
    embedder = get_fastembed(request.app)
    cohere = get_coherent(request.app)
    redis = Redis(redis=redis_client, cohere_client=cohere, session_id=session_id)
    llm = LLM()

    logger.info("Embedding query and retrieving relevant context and chat.")
    embedding = list(embedder.embed([query]))[0]
    relatedContext = redis.search(query, embedding, top_k=20, doc_type="chunk")
    relatedChat = redis.search(query, embedding, top_k=20, doc_type="message")
    print(relatedChat)
    contextChunk = None
    if chunkId is not None:
        contextChunk = redis.get_chunk_by_id(chunkId)

    messages = llm.format_query(query, relatedContext, relatedChat, contextChunk)

    async def streamer_generator():
        logger.info("Starting chat response streaming.")
        chatResponse = ""

        try:
            async for text in llm.stream(messages):
                chatResponse += text
                # Do NOT log or print individual loop items
                yield json.dumps({"type": "text", "content": text}) + "<END>"
                await asyncio.sleep(0)

            yield json.dumps({"type": "done"}) + "<END>"
            embeddings = list(embedder.embed([query, chatResponse]))
            redis.add_to_chat(
                [
                    {"text": query, "message_id": generateId()},
                    {"text": chatResponse, "message_id": generateId()},
                ],
                embeddings,
            )

        except Exception as e:
            logger.error(f"Error in streamer_generator: {str(e)}")
            yield json.dumps({"type": "error", "message": str(e)}) + "<END>"
            return

    return StreamingResponse(streamer_generator(), media_type="text/plain")


@safeExecution
async def generateImage(query: str, request: Request, chunkId: str | None):
    logger.info("generateImage endpoint called.")
    session_id = getattr(request.state, "session_id", None)
    user_id = getattr(request.state, "user_id", None)
    if user_id is None or session_id is None:
        logger.warning("Unauthorized call to chat")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized call to chat"
        )
    if chunkId is None:
        logger.warning("No chunkId provided to generateImage")
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
    logger.info("Generating image from prompt using FalAI.")
    image = falAi.generate_image(prompt)
    Image.open(image)
