
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import time

from src.inference.predict import predict_emotion
from src.llm.groq_client import explain_emotion

app = FastAPI(title="FER Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    try:
        start = time.time()
        result = predict_emotion(img)
        duration = time.time() - start

        return {
            "emotion": result["emotion"],
            "confidence": result["confidence"],
            "filename": file.filename,
            "latency_ms": round(duration * 1000, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ExplainRequest(BaseModel):
    emotion: str
    confidence: float


@app.post("/explain")
def explain(req: ExplainRequest):
    try:
        explanation = explain_emotion(req.emotion, req.confidence)
        return {"ai_explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        response = explain_emotion(req.message, 1.0)
        return {"reply": response}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}
