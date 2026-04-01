from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Depends
from pdfParser.parser import PDFParser
from Model.model import Model
from workers.modelWorker import model_worker
from workers.streamer import stream_output
from fastapi.responses import StreamingResponse
import asyncio


router = APIRouter()

BATCH_TOKEN_LIMIT = 3000


@router.post("/documents/upload-paper")
async def uploadPaper(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        input_q = asyncio.Queue()
        output_q = asyncio.Queue()
        parser = PDFParser(file_bytes)
        # model = Model("Qwen/Qwen2.5-1.5B-Instruct")
        model = Model("meta-llama/Meta-Llama-3-8B-Instruct")
        # model.initialize_model()

        asyncio.create_task(model_worker(input_q, output_q, model))

        async def handle_stream():
            for chunk in parser.stream():
                if not chunk:
                    continue

                if chunk["type"] == "noise":
                    continue

                elif chunk["type"] == "text":
                    # print("\nnew chunk pushing to input queue")
                    text = chunk["content"]
                    await input_q.put({"type": "text", "content": text})
                # elif chunk["type"] == "image":
                #     data = chunk["data"]
                #     await input_q.put({"type": "image", "content": data})
                # await asyncio.sleep(0)
            await input_q.put({"type": "end"})

        asyncio.create_task(handle_stream())

        return StreamingResponse(stream_output(output_q), media_type="text/plain")
    except Exception as e:
        print(e)
        return {"success": False}


import fitz

from fastapi import APIRouter, UploadFile, File
import fitz  # PyMuPDF
import json

@router.post("/generateDataset")
async def generateData(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        docs = fitz.open(stream=file_bytes, filetype="pdf")

        model = Model("meta-llama/Meta-Llama-3-8B-Instruct")

        dataset = [] 

        for page in docs:
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if block["type"] == 0:
                    text = " ".join(
                        " ".join(span["text"] for span in line["spans"])
                        for line in block["lines"]
                    ).strip()

                    if not text:
                        continue

                    # Call model
                    print(text)
                    response = await model.model_generate(text)
                    print(response)
                    if response is None:
                        continue
                    # Extract actual text from response
                    # try:
                    #     output_text = response.choices[0].message.content
                    # except Exception:
                    #     output_text = str(response)
                    output_text = response

                    # Build dataset entry
                    data = {
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert at simplifying complex research content for beginners. "
                                    "Your goal is to make the explanation easy to understand for a normal user "
                                    "with no technical background.\n\n"
                                    "Follow this structure strictly:\n\n"
                                    "1. Simplified Explanation:\n"
                                    "- Explain the idea in very simple language.\n"
                                    "- Use analogies or real-life examples when possible.\n"
                                    "- Avoid jargon. If needed, explain it in simple words.\n\n"
                                    "2. Key Points:\n"
                                    "- Provide 3–6 bullet points.\n"
                                    "- Keep them short and clear.\n\n"
                                    "3. Why It Matters:\n"
                                    "- Briefly explain why this concept is useful or important in real life.\n\n"
                                    "Rules:\n"
                                    "- Do NOT copy sentences from the input.\n"
                                    "- Do NOT use complex words unnecessarily.\n"
                                    "- Keep it concise but clear.\n"
                                    "- If the input is already simple, still format it in this structure."
                                ),
                            },
                            {"role": "user", "content": text},
                            {"role": "assistant", "content": output_text},
                        ]
                    }

                    dataset.append(data)

        # Save as JSONL (best for training)
        with open("dataset.jsonl", "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return {"status": "success", "samples_generated": len(dataset)}
    except Exception as e:
       print(e)