import json

async def stream_output(output_q):
    while True:
        result = await output_q.get()
        if not result or result is None:
            continue
        # print(result)
        yield json.dumps(result) + "<END>"
        output_q.task_done()
