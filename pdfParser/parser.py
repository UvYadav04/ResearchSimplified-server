import fitz
from .noise import is_noise
import asyncio
from utils.idGenerator import generateId
from controllers.chat import classifyQuery
import logging

logger = logging.getLogger("PDFParser")
logging.basicConfig(level=logging.INFO)


class PDFParser:
    def __init__(
        self,
        file_stream,
        redis,
        embedder=None,
    ):
        logger.info("Initializing PDFParser")
        self.doc = fitz.open(stream=file_stream, filetype="pdf")
        self.buffer = None
        self.batch = []
        self.token_count = 0
        self.embedder = embedder
        self.redis = redis

    def get_embeddings(self, chunks):
        logger.info("Getting embeddings for %d chunks", len(chunks))
        embeddings = list(self.embedder.embed(chunks))
        return embeddings

    async def _store_embeddings(self, chunks):
        print("storign len chunks : ", len(chunks))
        embeddings = self.get_embeddings([c["text"] for c in chunks])
        self.redis.add_chunks_batch(chunks, embeddings)

    def stream(self):
        logger.info("Starting PDF parsing and streaming chunks")
        chunks = []
        for index, page in enumerate(self.doc):
            blocks = page.get_text("dict")["blocks"]
            for block_idx, block in enumerate(blocks):
                # Do NOT log or print any loop item here
                result = self.process_block(block, index, block_idx)
                if result and result["type"] == "text":
                    chunks.append(
                        {"text": result["content"], "chunk_id": str(result["id"])}
                    )
                if result:
                    yield result

        logger.info("Finished iterating pages and blocks; embedding and storing chunks")
        asyncio.create_task(self._store_embeddings(chunks))
        logger.info("Yielding end marker for PDF stream")
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
        logger.info("Flushing PDFParser and returning end marker")
        return {"type": "end"}
