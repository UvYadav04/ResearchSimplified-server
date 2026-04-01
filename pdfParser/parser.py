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
        i = 0
        for page in self.doc:
            if i>4:
                break
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                result = self.process_block(block)

                if result: 
                    yield result

        yield {"type": "end"}  # simpler

    def process_block(self, block):
        if block["type"] == 0:
            text = " ".join(
                " ".join(span["text"] for span in line["spans"])
                for line in block["lines"]
            ).strip()

            if is_noise(text):
                return {"type": "noise", "content": None}

            if text:
                return {"type": "text", "content": text}

        if block["type"] == 1:
            data = block["image"]
            return {"type": "image", "data":data, "size": len(data)}
        return None

    def flush(self):
        return {"type": "end"}
