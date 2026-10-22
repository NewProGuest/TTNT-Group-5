import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": not service.fallback_mode,
        "device": str(service.device),
        "model_name": service.model_name,
    }


@app.post("/predict")
def predict(payload: PredictionRequest) -> Dict[str, Any]:
    try:
        result = service.predict(payload.text)
        return result
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", "8000"))
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)
