import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.model_service import FakeNewsModelService


class PredictionRequest(BaseModel):
    text: str


app = FastAPI(title="Fake News Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = FakeNewsModelService(model_path=os.getenv("MODEL_PATH"))


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Fake News Detection API is running."}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": not service.fallback_mode,
        "device": str(service.device),
        "model_name": service.model_name,
        "fallback": service.fallback_mode,
    }


@app.post("/predict")
def predict(payload: PredictionRequest) -> Dict[str, Any]:
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Văn bản đầu vào không được rỗng.")

    result = service.predict(text)
    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)
