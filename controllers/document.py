from fastapi import UploadFile, File, Request, HTTPException, status
import asyncio
from workers.modelWorker import model_worker
from workers.streamer import stream_output
from startupFunctions import get_mongo, get_client
from fastapi.responses import StreamingResponse
import asyncio
from PIL import Image
import os
from pdfParser.parser import PDFParser
from Model.model import Model


async def uploadPaper(request: Request, file):
    try:
        user_id = getattr(request.state, "user_id", None)
        mongo = get_mongo(request.app)

        if mongo is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Can't upload the paper at the moment.",
            )

        user_db = mongo.get_collection("users")

        userInfo = None
        if user_id:
            userInfo = user_db.find_one({"_id": user_id})   # ✅ FIX

        if userInfo and userInfo.get("documentUploads", 0) >= int(os.environ["USER_DOCS_ALLOWED"]): 
            raise HTTPException(
                status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                detail="You can upload only 5 paper in free tier.",
            )

        file_bytes = await file.read()

        input_q = asyncio.Queue()
        output_q = asyncio.Queue()

        parser = PDFParser(file_bytes)

        hf_inference = await get_client(request.app)

        model = Model(hf_inference)

        asyncio.create_task(model_worker(input_q, output_q, model))

        async def handle_stream():
            for chunk in parser.stream():
                if not chunk:
                    continue

                if chunk["type"] == "noise":
                    continue

                elif chunk["type"] == "text":
                    page = chunk["page"]
                    if user_id is None and page >= 1:
                        break
                    text = chunk["content"]
                    await input_q.put(
                        {
                            "type": "text",
                            "content": text,
                            "page": chunk["page"],
                            "block_idx": chunk["block_idx"],
                            "id": chunk["id"],
                        }
                    )
                # elif chunk["type"] == "image":
                #     data = chunk["data"]
                #     image = Image.open(io.BytesIO(data))
                #     image.show()
                #     await input_q.put({"type": "image", "content": data})
                await asyncio.sleep(0)
            await input_q.put({"type": "end"})

        asyncio.create_task(handle_stream())

        # if user_db and userInfo:
        #     user_db.find_one_and_update({"_id": id}, {"$inc": {"documentUploads": 1}})
        return StreamingResponse(stream_output(output_q), media_type="text/plain")
    except Exception as e:
        print("I am error : ",e)
        await input_q.put({"type": "error","message": str(e)})
        return {"success": False}
