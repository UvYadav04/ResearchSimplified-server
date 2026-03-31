import asyncio
from Model.model import Model


async def model_worker(input_q: asyncio.Queue, output_q: asyncio.Queue, model: Model):
    while True:
        top = await input_q.get()
        generator = model.stream_generate(top)
        for token in generator:
            await output_q.put(token)
        input_q.task_done()
