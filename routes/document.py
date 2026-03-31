from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Depends
from pdfParser.parser import PDFParser, getPdfParser
from modelBucket.bucket import Bucket
from Model.model import Model
from Queue.queue import Queue
from workers.modelWorker import model_worker
from workers.streamer import stream_output
from fastapi.responses import StreamingResponse
import asyncio


router = APIRouter()


BATCH_TOKEN_LIMIT = 3000

@router.post("/upload-paper")
async def uploadPaper(file: UploadFile = File(...)):

    input_q = asyncio.Queue()
    output_q = asyncio.Queue()

    parser = PDFParser(file.file)
    model = Model()
    bucket = Bucket(max_tokens=8000, queue=input_q)

    asyncio.create_task(model_worker(model, input_q, output_q))

    async def handle_stream():
        for chunk in parser.stream():
            if not chunk:
                continue

            if chunk["type"] == "noise":
                continue

            elif chunk["type"] == "text":
                formatted = model.apply_chat_template(chunk["content"])
                await bucket.add_to_bucket(formatted, formatted)

            elif chunk["type"] == "image":
                pass

        await bucket.flush()

    asyncio.create_task(handle_stream())

    return StreamingResponse(stream_output(output_q), media_type="text/plain")
