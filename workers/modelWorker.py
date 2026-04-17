import asyncio
from Model.model import Model
from SafeExecution.safeExecution import safeExecution
from Model.llm import LLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("modelWorker")


@safeExecution
async def model_worker(input_q: asyncio.Queue, output_q: asyncio.Queue, model: Model):
    lastChunk = None
    llm = LLM()
    logger.info("Model worker started.")
    while True:
        top = await input_q.get()
        logger.info(f"Received from input_q")
        if top is None:
            logger.warning("Received None from input_q. Skipping iteration.")
            continue

        if top["type"] == "error":
            logger.error(f"Error received in input_q: {top.get('message')}")
            await output_q.put({"type": "end", "message": top["message"]})
            input_q.task_done()
            break
        elif top["type"] == "end":
            logger.info("Received end signal. Shutting down model_worker.")
            await output_q.put({"type": "end"})
            input_q.task_done()
            break

        else:
            logger.info(
                f"Processing block - page: {top.get('page')}, block: {top.get('block_idx')}, id: {top.get('id')}"
            )
            await output_q.put(
                {
                    "type": "originalContent",
                    "content": top["content"],
                    "page": top["page"],
                    "content-type": top["type"],
                    "block_idx": top["block_idx"],
                    "id": str(top["id"]),
                }
            )
            await asyncio.sleep(0)
            messages = llm.format_instruction(top["content"], lastChunk)
            if messages is None:
                logger.warning("Messages is None after formatting. Skipping iteration.")
                continue

            try:
                async for chunk in llm.stream(messages):
                    token = chunk

                    if token:
                        await output_q.put({"type": "text", "content": token})
                        await asyncio.sleep(1)

                logger.info("Finished streaming tokens for this chunk.")
                await output_q.put({"type": "done"})

            except Exception as e:
                logger.error(f"Exception during LLM streaming: {e}")
                await output_q.put({"type": "error", "content": str(e)})
                return

            finally:
                logger.info("Marking input_q.task_done() for current item.")
                input_q.task_done()

            if "content" in top:
                lastChunk = top["content"]
                logger.info("Updated lastChunk content.")


@safeExecution
def extract_token(chunk):
    if hasattr(chunk, "text") and chunk.text:
        logger.debug("Extracted token from .text attribute.")
        return chunk.text
    try:
        result = chunk.choices[0].delta.content
        logger.debug("Extracted token from .choices[0].delta.content.")
        return result
    except (AttributeError, IndexError, KeyError) as e:
        logger.warning(f"Failed to extract token: {e}")
        return None
