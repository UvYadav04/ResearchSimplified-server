import asyncio
from Model.model import Model


async def model_worker(input_q: asyncio.Queue, output_q: asyncio.Queue, model: Model):
    lastChunk = None
    while True:
        top = await input_q.get()
        # print(top["type"])

        if top["type"] != "end":
            await output_q.put({"type": "originalContent", "content": top["content"]})
            await asyncio.sleep(0)

        generator = model.stream_generate(top,lastChunk)
        if "content" in top:
            lastChunk = top["content"]
        if generator is None:
            continue
        try:
            for chunk in generator:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                if not choice.delta:
                    continue

                token = choice.delta.content

                if token:
                    await output_q.put({"type": "text", "content": token})
                    await asyncio.sleep(0)
            await output_q.put({"type": "end"})
            await asyncio.sleep(0)
        except Exception as e:
            print("Streaming failed:", e)
        input_q.task_done()
