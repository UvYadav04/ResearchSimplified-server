import asyncio
from Model.model import Model


async def model_worker(queue: asyncio.queue, model: Model):
    while True:
        top = await queue.get()
        if top["type"] == "end":
            queue.task_done()
        elif top["type"] == "text":
            res = await model.generate(top["tokens"])
        elif top["type"] == "image":
            res = await model.generate(top["image"])
