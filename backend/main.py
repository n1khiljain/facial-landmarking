"""
Backend API - Receives image, processes it, returns result
"""
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import sys

# Import face_landmarker from parent directory
sys.path.insert(0, "/Users/nikhiljain/Desktop/Coding Projects/face-recognition-ai")
from face_landmarker import process_image

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze")
async def analyze(file: UploadFile):
    """
    POST an image file -> returns the image with face landmarks drawn
    """
    try:
        image_bytes = await file.read()
        result_bytes = process_image(image_bytes)
        return Response(content=result_bytes, media_type="image/jpeg")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
