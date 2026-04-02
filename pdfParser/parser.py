import fitz
from .noise import is_noise

import asyncio
class PDFParser:
    def __init__(self, file_stream):
        self.doc = fitz.open(stream=file_stream, filetype="pdf")

        self.buffer = None
        self.batch = []
        self.token_count = 0

    def stream(self):
        for index, page in enumerate(self.doc):
            blocks = page.get_text("dict")["blocks"]
            for block_idx, block in enumerate(blocks):
                print(f"parser : page : {index} block: {block_idx}")
                result = self.process_block(block,index,block_idx)

                if result: 
                    yield result

        yield {"type": "end"}  # simpler

    def process_block(self, block,index,block_idx):
        if block["type"] == 0:
            text = " ".join(
                " ".join(span["text"] for span in line["spans"])
                for line in block["lines"]
            ).strip()

            if is_noise(text):
                return {"type": "noise", "content": None}

            if text:
                return {"type": "text", "content": text,"page":index,"block_idx":block_idx}

        if block["type"] == 1:
            data = block["image"]
            return {"type": "image", "data":data, "size": len(data),"page":index,"block_idx":block_idx}
        return None

    def flush(self):
        return {"type": "end"}
