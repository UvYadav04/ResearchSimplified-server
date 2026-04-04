import asyncio
from Model.model import Model
from SafeExecution.safeExecution import safeExecution


@safeExecution
async def model_worker(input_q: asyncio.Queue, output_q: asyncio.Queue, model: Model):
    lastChunk = None
    while True:
        top = await input_q.get()

        if top["type"] == "error":
            await output_q.put({"type": "end", "message": top["message"]})
            input_q.task_done()
            break
        elif top["type"] == "end":
            await output_q.put({"type": "end"})
            input_q.task_done()
            break

        else:
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
            print(f"modelworker : page:{top["page"]} block:{top["block_idx"]}")
            await asyncio.sleep(0)

        generator = model.stream_document(top, lastChunk)
        await output_q.put({"type": "error", "message": "Generator is not working today"})
        await asyncio.sleep(0)
        generator.close()

        if generator is None:
            continue
        # elif type(generator) == dict and "error" in generator:
        else:
            try:
                for chunk in generator:
                    token = extract_token(chunk)
                    if token:
                        if "same" in token or "Same" in token:
                            await output_q.put({"type": "sameContent", "content": None})
                            await asyncio.sleep(0)
                            break
                        await output_q.put({"type": "text", "content": token})
                        await asyncio.sleep(0)
                await asyncio.sleep(0)
            except Exception as e:
                print("Streaming failed:", e)
            finally:
                if hasattr(generator, "close"):
                    generator.close()
        input_q.task_done()
        if "content" in top:
            lastChunk = top["content"]


@safeExecution
def extract_token(chunk):
    if hasattr(chunk, "text") and chunk.text:
        return chunk.text
    try:
        return chunk.choices[0].delta.content
    except (AttributeError, IndexError, KeyError):
        return None
