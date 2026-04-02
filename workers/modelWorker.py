import asyncio
from Model.model import Model


async def model_worker(input_q: asyncio.Queue, output_q: asyncio.Queue, model: Model):
    lastChunk = None
    while True:
        top = await input_q.get()
        print(top["type"])

        if top["type"] != "end":
            await output_q.put({"type": "originalContent", "content": top["type"]})
            await asyncio.sleep(0)

        generator = model.stream_generate(top, lastChunk)
        if "content" in top:
            lastChunk = top["content"]
        if generator is None:
            continue
        try:
            for chunk in generator:
                token = extract_token(chunk)
                if token:
                    await output_q.put({"type": top["type"], "content": token})
                    await asyncio.sleep(0)
            await output_q.put({"type": "end"})
            await asyncio.sleep(0)
        except Exception as e:
            print("Streaming failed:", e)
        input_q.task_done()


def extract_token(chunk):
    if hasattr(chunk, "text") and chunk.text:
        return chunk.text

    try:
        return chunk.choices[0].delta.content
    except (AttributeError, IndexError, KeyError):
        return None
