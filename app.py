import os
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-zh-v1.5")
DEVICE = os.getenv("DEVICE", "cpu")

app = FastAPI(title="BGE Small ZH API", version="1.0.0")
model: Optional[SentenceTransformer] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model() -> None:
    global model
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.options("/v1/embeddings")
def embeddings_options() -> Response:
    return Response(status_code=204)


@app.post("/v1/embeddings")
def embeddings(payload: dict[str, Any]) -> JSONResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if "model" in payload and payload.get("model") != MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {payload.get('model')}")

    texts: Optional[List[str]] = None
    if isinstance(payload.get("input"), list):
        texts = payload.get("input")
    elif isinstance(payload.get("input"), str):
        texts = [payload.get("input")]
    elif isinstance(payload.get("texts"), list):
        texts = payload.get("texts")
    elif isinstance(payload.get("text"), str):
        texts = [payload.get("text")]

    if not texts:
        raise HTTPException(status_code=400, detail="Provide 'input', 'text', or 'texts'")

    normalize = bool(payload.get("normalize", True))
    vectors = model.encode(texts, normalize_embeddings=normalize)
    data = [
        {"object": "embedding", "index": i, "embedding": emb.tolist()}
        for i, emb in enumerate(vectors)
    ]
    result = {
        "object": "list",
        "model": MODEL_NAME,
        "data": data,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
    return JSONResponse(result)
