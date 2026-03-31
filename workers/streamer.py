async def stream_output(output_q):
    while True:
        result = await output_q.get()
        yield result
        output_q.task_done()
