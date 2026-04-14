from fastapi import Request, HTTPException, status, Response
from startupFunctions import get_mongo
from fastapi.responses import JSONResponse
import jwt
import os
from SafeExecution.safeExecution import safeExecution
from bson import ObjectId
import logging

logger = logging.getLogger("userController")
logging.basicConfig(level=logging.INFO)


@safeExecution
async def getUserInfo(request: Request):
    try:
        logger.info("getUserInfo endpoint called")
        mongo = get_mongo(request.app)
        if mongo is None:
            logger.error("Mongo connection is None in getUserInfo")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get info at the moment",
            )
        user_id = getattr(request.state, "user_id", None)
        user_col = mongo.get_collection("users")

        if not user_id:
            logger.info("No user_id found in request.state for getUserInfo")
            return JSONResponse({"success": True, "userInfo": None})

        # Not logging any loop item here
        userInfo = user_col.find_one({"_id": ObjectId(user_id)})

        if userInfo is None:
            logger.warning(f"User info not found for user_id: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized"
            )
        userInfo["_id"] = str(userInfo["_id"])
        logger.info(f"User info successfully retrieved for user_id: {user_id}")
        return JSONResponse({"success": True, "userInfo": userInfo})
    except Exception as e:
        logger.error(f"Exception in getUserInfo: {e}")


@safeExecution
async def login(request: Request, response: Response):
    logger.info("login endpoint called")
    mongo = get_mongo(request.app)
    if mongo is None:
        logger.error("Mongo connection is None during login")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to login at the moment",
        )
    user_col = mongo.get_collection("users")
    if user_col is None:
        logger.error("user_col is None during login")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to login at the moment",
        )
    body = await request.json()
    email = body["email"]
    name = body["name"]
    logger.info(f"Login attempt for email: {email}")
    userDetails = user_col.find_one({"email": email})

    if userDetails is None:
        logger.info(f"New user detected, creating user for email: {email}")
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
        logger.info(f"Existing user logging in, email: {email}")
        user_id = userDetails["_id"]

    jwt_token = jwt.encode(
        {"user_id": str(user_id)},
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

    logger.info(f"User {email} logged in and JWT cookie set")
    return response


@safeExecution
async def logout(request: Request, response: Response):
    logger.info("logout endpoint called")
    response = JSONResponse({"success": True, "message": "Logged in Successfully"})
    response.delete_cookie(
        key="research_simplified",
        secure=True,
        samesite="none",
        httponly=True,
        path="/",
    )
    logger.info("JWT cookie deleted and logout response returned")
    return response
