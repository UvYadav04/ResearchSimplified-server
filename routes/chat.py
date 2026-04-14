from fastapi import APIRouter, Request, Response
from controllers.chat import classifyQuery, handleQuery, generateImage
from startupFunctions import get_gemini, get_client, get_mongo
from fastapi.responses import StreamingResponse
from fastapi import status, HTTPException
import json
from SafeExecution.safeExecution import safeExecution
from bson import ObjectId
import os

router = APIRouter()


@router.post("/chat/query")
async def handleChat(request: Request, response: Response):
    try:
        body = await request.json()
        query = body["query"]
        chunkId = None
        mongo = get_mongo(request.app)
        user_db = mongo.get_collection("users")

        if "chunkId" in body:
            chunkId = body["chunkId"]
        user_id = getattr(request.state, "user_id", None)
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Please login to send a query.",
            )
        obectified_id = None
        if user_id:
            obectified_id = ObjectId(user_id)
        userInfo = None
        if obectified_id:
            userInfo = user_db.find_one({"_id": obectified_id})

        if userInfo and userInfo.get("chatCounts", 0) >= int(
            os.environ["USER_CHATS_ALLOWED"]
        ):
            raise HTTPException(
                status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                detail="You can do only 7 queries for each document in free tier.",
            )
        
        if user_db is not None and userInfo:
            user_db.find_one_and_update(
                {"_id": obectified_id},
                {"$inc": {"chatCounts": 1},},
            )

        return await handleQuery(query, request, chunkId)
    except Exception as e:
        print(e)
        return json.dumps({"type": "error", "message": str(e)}) + "<END>"
