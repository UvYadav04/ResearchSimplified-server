from fastapi import UploadFile, File, Request, HTTPException, status
import asyncio
from workers.modelWorker import model_worker
from workers.streamer import stream_output
from startupFunctions import (
    get_mongo,
    get_client,
    get_fastembed,
    get_redis,
    get_groq,
    get_coherent,
)
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
from Redis.redis import Redis
from controllers.chat import classifyQuery
from PIL import Image
import os
from pdfParser.parser import PDFParser
from Model.model import Model
from bson import ObjectId
from SafeExecution.safeExecution import safeExecution


@safeExecution
async def uploadPaper(request: Request, file):
    try:
        user_id = getattr(request.state, "user_id", None)
        session_id = getattr(request.state, "session_id", None)
        mongo = get_mongo(request.app)
        embedder = get_fastembed(request.app)
        redis_client = get_redis(request.app)
        cohere = get_coherent(request.app)
        redis = Redis(redis=redis_client, cohere_client=cohere, session_id=session_id)

        if mongo is None or embedder is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Can't upload the paper at the moment.",
            )

        user_db = mongo.get_collection("users")
        obectified_id = None
        if user_id:
            obectified_id = ObjectId(user_id)
        userInfo = None
        if obectified_id:
            userInfo = user_db.find_one({"_id": obectified_id})  # ✅ FIX

        if userInfo and userInfo.get("documentUploads", 0) >= 2:
            raise HTTPException(
                status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                detail="You can upload only 2 paper in free tier.",
            )

        file_bytes = await file.read()

        input_q = asyncio.Queue()
        output_q = asyncio.Queue()

        parser = PDFParser(file_bytes, redis, embedder)

        hf_inference = await get_client(request.app)
        # groq = await get_groq(request.app)

        model = Model(hf_inference)

        asyncio.create_task(model_worker(input_q, output_q, model))
        chunks = []

        async def handle_stream():
            for chunk in parser.stream():
                if not chunk:
                    continue

                if chunk["type"] == "noise":
                    continue

                elif chunk["type"] == "text":
                    page = chunk["page"]
                    if (obectified_id is None and page >= 1) or page >= 3:
                        break
                    text = chunk["content"]

                    options = (
                        "Core content (methods, results, explanation, concepts)",
                        "Non-core content (authors, references, metadata, acknowledgments)",
                    )
                    response = await classifyQuery(
                        text,
                        options=[
                            "Core content (methods, results, explanation, concepts)",
                            "Non-core content (authors, references, metadata, acknowledgments)",
                        ],
                    )
                    query_type = response[0]["label"]
                    if query_type == options[1]:
                        continue
                    chunks.append(
                        {"text": chunk["content"], "chunk_id": str(chunk["id"])}
                    )
                    await input_q.put(
                        {
                            "type": "text",
                            "content": text,
                            "page": chunk["page"],
                            "block_idx": chunk["block_idx"],
                            "id": chunk["id"],
                        }
                    )
                    await asyncio.sleep(0)
                # elif chunk["type"] == "image":
                #     data = chunk["data"]
                #     image = Image.open(io.BytesIO(data))
                #     image.show()
                #     await input_q.put({"type": "image", "content": data})
            await input_q.put({"type": "end"})

        asyncio.create_task(handle_stream())
        if user_db is not None and userInfo:
            user_db.find_one_and_update(
                {"_id": obectified_id},
                {"$inc": {"documentUploads": 1}, "$set": {"chatCounts": 0}},
            )
        embeddings = parser.get_embeddings([chunk["text"] for chunk in chunks])
        redis.add_chunks_batch(chunks, embeddings)
        return StreamingResponse(stream_output(output_q), media_type="text/plain")
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"success": False, "message": str(e)}
        )
