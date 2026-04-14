import asyncio
from Model.model import Model
from SafeExecution.safeExecution import safeExecution
from Model.llm import LLM


@safeExecution
async def model_worker(input_q: asyncio.Queue, output_q: asyncio.Queue, model: Model):
    lastChunk = None
    llm = LLM()
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
            messages = llm.format_instruction(top["content"], lastChunk)

            try:
                async for chunk in llm.stream(messages):
                    token = extract_token(chunk)

                    if token:
                        await output_q.put({"type": "text", "content": token})

                await output_q.put({"type": "done"})

            except Exception as e:
                await output_q.put({"type": "error", "content": str(e)})
                return

            finally:
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
