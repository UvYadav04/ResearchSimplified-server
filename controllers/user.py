from fastapi import Request, HTTPException, status, Response
from startupFunctions import get_mongo
from fastapi.responses import JSONResponse
import jwt
import os
from SafeExecution.safeExecution import safeExecution
from bson import ObjectId


@safeExecution
async def getUserInfo(request: Request):
    try:
        mongo = get_mongo(request.app)
        if mongo is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get info at the moment",
            )
        user_id = getattr(request.state, "user_id", None)
        user_col = mongo.get_collection("users")

        if not user_id:
            return JSONResponse({"success": True, "userInfo": None})

        allUsers = user_col.find_one({})

        userInfo = user_col.find_one({"_id": ObjectId(user_id)})

        if userInfo is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized"
            )
        userInfo["_id"] = str(userInfo["_id"])
        return JSONResponse({"success": True, "userInfo": userInfo})
    except Exception as e:
        print(e)


@safeExecution
async def login(request: Request, response: Response):
    mongo = get_mongo(request.app)
    if mongo is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to login at the moment",
        )
    user_col = mongo.get_collection("users")
    if user_col is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to login at the moment",
        )
    body = await request.json()
    email = body["email"]
    name = body["name"]
    userDetails = user_col.find_one({"email": email})

    if userDetails is None:
        newUser = user_col.insert_one(
            {
                "name": name,
                "email": email,
                "chatCounts": 0,
                "documentUploads": 0,
                "imageGenerations": 0,
            }
        )
        user_id = newUser.inserted_id
    else:
        user_id = userDetails["_id"]

    jwt_token = jwt.encode(
        {"user_id": str(user_id)},  # convert ObjectId to string
        os.environ.get("JWT_SECRET"),
        algorithm="HS256",
    )

    response = JSONResponse({"success": True, "message": "Logged in Successfully"})

    response.set_cookie(
        key="research_simplified",
        value=jwt_token,
        expires=60 * 60 * 24 * 7,
        secure=True,
        samesite="none",
        httponly=True,
        path="/",
    )

    return response


@safeExecution
async def logout(request: Request, response: Response):
    response = JSONResponse({"success": True, "message": "Logged in Successfully"})
    response.delete_cookie(
        key="research_simplified",
        secure=True,
        samesite="none",
        httponly=True,
        path="/",
    )
    return response
