import fitz
from .noise import is_noise
import asyncio
from utils.idGenerator import generateId
from controllers.chat import classifyQuery


class PDFParser:
    def __init__(
        self,
        file_stream,
        redis,
        embedder=None,
    ):
        self.doc = fitz.open(stream=file_stream, filetype="pdf")
        self.buffer = None
        self.batch = []
        self.token_count = 0
        self.embedder = embedder
        self.redis = redis

    def get_embeddings(self, chunks):
        embeddings = list(self.embedder.embed(chunks))
        return embeddings

    def stream(self):
        chunks = []
        for index, page in enumerate(self.doc):
            blocks = page.get_text("dict")["blocks"]
            for block_idx, block in enumerate(blocks):
                # print(f"parser : page : {index} block: {block_idx}")
                result = self.process_block(block, index, block_idx)
                if result["type"] == "text":
                    chunks.append(
                        {"text": result["content"], "chunk_id": str(result["id"])}
                    )
                if result:
                    yield result

        embeddings = self.get_embeddings([chunk["text"] for chunk in chunks])
        self.redis.add_chunks_batch(chunks, embeddings)
        yield {"type": "end"}  # simpler

    def process_block(self, block, index, block_idx):
        if block["type"] == 0:
            text = " ".join(
                " ".join(span["text"] for span in line["spans"])
                for line in block["lines"]
            ).strip()

            if is_noise(text):
                return {"type": "noise", "content": None}

            if text:
                return {
                    "type": "text",
                    "content": text,
                    "page": index,
                    "block_idx": block_idx,
                    "id": generateId(),
                }

        if block["type"] == 1:
            data = block["image"]
            return {
                "type": "image",
                "data": data,
                "size": len(data),
                "page": index,
                "block_idx": block_idx,
                "id": generateId(),
            }
        return None

    def flush(self):
        return {"type": "end"}
