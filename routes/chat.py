from fastapi import APIRouter, Request, Response
from controllers.chat import classifyQuery
from startupFunctions import get_gemini, get_client

router = APIRouter()


@router.post("/query")
async def handleChat(request: Request, response: Response):
    body = await request.json()
    query = body["query"]
    chunkId = None
    if "chunkId" in body:
        chunkId = body["chunkId"]
    if "messages" in body:
        messages = body["messages"]
    client = get_gemini(request.app)
    # we need to classify the query type here as : "sub-part query" | "summarize" | "image generation" | "follow-up"
    response = await classifyQuery(query)
    query_type = response[0]["label"]

    match query_type:
        case "A standalone general query":
            return {"type": "normal_query"}
        case "A request to summarize content":
            return {"type": "summarization"}
        case "A follow-up query depending on previous context":
            return {"type": "follow_up"}
        case "A request to generate an image":
            return {"type": "image_generation"}