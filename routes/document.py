from fastapi import APIRouter, HTTPException, Request, UploadFile, File,Depends
from pdfParser.parser import PDFParser,getPdfParser
from modelBucket.bucket import Bucket
from Model.model import Model
from Queue.queue import Queue
from workers.modelWorker import model_worker
import asyncio
router = APIRouter()


BATCH_TOKEN_LIMIT = 3000


@router.post("/upload-paper")
async def uploadPaper(request: Request, file: UploadFile = File(...)):
    parser = PDFParser(file.file)
    textBucket = Bucket()
    modelManager = Model()
    messageQueue = asyncio.queue
    asyncio.create_task(model_worker(messageQueue,modelManager))
    
    async def handleStream():
        for chunk in parser.stream():
        if chunk:
            type = chunk["type"]
            if type == "noise":
                continue
            elif type == "text":
                tokens = modelManager.apply_chat_template(chunk)
                textBucket.add_to_bucket(tokens)
            elif type == "image":
                #convert image to tokens
    asyncio.create_task(handleStream())
    return {"message": "Processed"}
