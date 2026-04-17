import json
from SafeExecution.safeExecution import safeExecution


@safeExecution
async def stream_output(output_q):
    while True:
        result = await output_q.get()
        if not result or result is None:
            continue
        if result["type"] == "end" or result["type"] == "error":
            yield json.dumps(result)
            break
        print("streaming : ", result)
        yield json.dumps(result) + "<END>"
        output_q.task_done()
