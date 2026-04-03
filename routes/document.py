from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Depends


from controllers.document import uploadPaper

router = APIRouter()

BATCH_TOKEN_LIMIT = 3000


@router.post("/documents/upload-paper")
async def handle_upload_paper(request: Request, file: UploadFile = File(...)):
    return await uploadPaper(request,file)
