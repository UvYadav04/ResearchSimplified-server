from fastapi import APIRouter, Request,Response
from controllers.user import getUserInfo,login,logout
router = APIRouter()


@router.get("/user-info")
async def get_user_info(request: Request):
    return await getUserInfo(request)

@router.post("/login")
async def login_user(request:Request,response:Response):
    return await login(request,response)

@router.post("/logout")
async def login_user(request:Request,response:Response):
    return await logout(request,response)