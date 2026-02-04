import os
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

from item_search import Item, ItemSearchEngine, SearchConfig, SearchRequest as EngineSearchRequest
from item_search.models import Heuristics, Thresholds, Weights

MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-zh-v1.5")
DEVICE = os.getenv("DEVICE", "cpu")

app = FastAPI(title="BGE Small ZH API", version="1.0.0")
model: Optional[SentenceTransformer] = None
engine: ItemSearchEngine | None = None
_DEMO_DIR = Path(__file__).resolve().parent / "demo"
if _DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(_DEMO_DIR), html=True), name="demo")


@app.get("/")
def demo_index() -> RedirectResponse:
    if not _DEMO_DIR.exists():
        raise HTTPException(status_code=404, detail="demo not found")
    return RedirectResponse(url="/demo/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model() -> None:
    global model, engine
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    engine = ItemSearchEngine(model)


@app.get("/health")
def health() -> dict[str, str]:
    items = engine.loaded_item_count() if engine is not None else 0
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE, "items_loaded": str(items)}


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


class ItemIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    type: str | list[str] | None = None
    desc_labels: list[str] = Field(default_factory=list, alias="labels")


class LoadItemsIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[ItemIn]
    enable_bm25: bool = True


class ThresholdsIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    accept_score: float = 0.55
    clarify_score: float = 0.45

    accept_p_nn: float = 0.01
    clarify_p_nn: float = 0.05
    accept_p_jj: float = 0.001
    clarify_p_jj: float = 0.01
    tau_cover: float = 0.35
    min_coverage: float = 0.5
    min_coverage_no_nn: float = 0.7
    tau_margin: float = 0.15
    tau_margin_no_nn: float = 0.2


class WeightsIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    alpha_nn: float = 0.7
    beta_jj: float = 0.3
    gamma_bm25: float = 0.1


class HeuristicsIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    split_compounds: bool = True
    split_max_len: int = Field(default=3, ge=2, le=12)
    head_nouns: list[str] = Field(default_factory=lambda: ["车", "门", "箱", "塔"])

    nn_suffix_match: bool = True
    nn_suffix_boost_to: float = Field(default=0.95, ge=0.0, le=1.0)


class SearchConfigIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    top_k: int = 5
    recall_topn_bm25: int = 50
    recall_topn_name_vec: int = 50
    recall_topm_desc_label: int = 50
    thresholds: ThresholdsIn = Field(default_factory=ThresholdsIn)
    weights: WeightsIn = Field(default_factory=WeightsIn)
    heuristics: HeuristicsIn = Field(default_factory=HeuristicsIn)
    enable_bm25: bool = True


class ItemSearchIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    candidate_ids: list[str] | None = None
    config: SearchConfigIn = Field(default_factory=SearchConfigIn)
    debug: bool = False
    pos_backend: str = "jieba"


class ScoreTopNIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    query: str
    item_ids: list[str] = Field(default_factory=list, alias="candidate_ids")
    top_n: int = Field(default=10, ge=1)
    debug: bool = False
    config: SearchConfigIn = Field(default_factory=SearchConfigIn)
    pos_backend: str = "jieba"


@app.post("/v1/items/load")
def load_items(payload: LoadItemsIn) -> JSONResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    items = [
        Item(
            id=i.id,
            name=i.name,
            aliases=tuple(i.aliases),
            type=i.type,
            desc_labels=tuple(i.desc_labels),
        )
        for i in payload.items
    ]

    result = engine.load_items(items, enable_bm25=payload.enable_bm25)
    return JSONResponse(
        {
            "ok": True,
            "item_count": result.item_count,
            "name_phrase_count": result.name_phrase_count,
            "desc_label_count": result.desc_label_count,
            "neg_name_samples": result.neg_name_samples,
            "neg_desc_samples": result.neg_desc_samples,
        }
    )


@app.post("/v1/item_search")
def item_search(payload: ItemSearchIn) -> JSONResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    cfg = SearchConfig(
        top_k=payload.config.top_k,
        recall_topn_bm25=payload.config.recall_topn_bm25,
        recall_topn_name_vec=payload.config.recall_topn_name_vec,
        recall_topm_desc_label=payload.config.recall_topm_desc_label,
        thresholds=Thresholds(**payload.config.thresholds.model_dump()),
        weights=Weights(**payload.config.weights.model_dump()),
        heuristics=Heuristics(**{**payload.config.heuristics.model_dump(), "head_nouns": tuple(payload.config.heuristics.head_nouns)}),
        enable_bm25=payload.config.enable_bm25,
    )
    req = EngineSearchRequest(
        query=payload.query,
        config=cfg,
        debug=payload.debug,
        candidate_ids=tuple(payload.candidate_ids) if payload.candidate_ids is not None else None,
        pos_backend=payload.pos_backend,
    )

    try:
        res = engine.search(req)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    def _item_out(it):
        if it is None:
            return None
        out = {"id": it.id, "name": it.name, "score": it.score, "confidence": it.confidence}
        if it.explain is not None:
            out["explain"] = {
                "s_total": it.explain.s_total,
                "s_nn": it.explain.s_nn,
                "s_nn_vec": it.explain.s_nn_vec,
                "s_nn_bm25": it.explain.s_nn_bm25,
                "s_jj": it.explain.s_jj,
                "coverage": it.explain.coverage,
                "margin_ratio": it.explain.margin_ratio,
                "p_nn": it.explain.p_nn,
                "p_jj": it.explain.p_jj,
                "matched_nn": it.explain.matched_nn,
                "matched_jj": it.explain.matched_jj,
            }
        return out

    return JSONResponse(
        {
            "decision": {"status": res.decision.status, "reason": res.decision.reason},
            "parsed": {
                "raw": res.parsed.raw,
                "nn": list(res.parsed.nn),
                "jj": list(res.parsed.jj),
                "head_noun": res.parsed.head_noun,
                "tokens": [{"text": t.text, "pos": t.pos} for t in res.parsed.tokens],
            },
            "best": _item_out(res.best),
            "alternatives": [_item_out(a) for a in res.alternatives],
        }
    )


@app.post("/v1/item_search/topn")
def item_search_topn(payload: ScoreTopNIn) -> JSONResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine not ready")

    item_ids = [str(v) for v in (payload.item_ids or []) if str(v).strip()]
    top_n = int(payload.top_n)
    if top_n <= 0:
        raise HTTPException(status_code=400, detail="top_n must be >= 1")
    if not item_ids:
        return JSONResponse({"items": []})

    cfg = SearchConfig(
        top_k=top_n,
        recall_topn_bm25=payload.config.recall_topn_bm25,
        recall_topn_name_vec=payload.config.recall_topn_name_vec,
        recall_topm_desc_label=payload.config.recall_topm_desc_label,
        thresholds=Thresholds(**payload.config.thresholds.model_dump()),
        weights=Weights(**payload.config.weights.model_dump()),
        heuristics=Heuristics(**{**payload.config.heuristics.model_dump(), "head_nouns": tuple(payload.config.heuristics.head_nouns)}),
        enable_bm25=payload.config.enable_bm25,
    )
    req = EngineSearchRequest(
        query=payload.query,
        config=cfg,
        debug=payload.debug,
        candidate_ids=tuple(item_ids),
        pos_backend=payload.pos_backend,
    )

    try:
        res = engine.search(req)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    def _item_out(it):
        if it is None:
            return None
        out = {"id": it.id, "name": it.name, "score": it.score, "confidence": it.confidence}
        if it.explain is not None:
            out["explain"] = {
                "s_total": it.explain.s_total,
                "s_nn": it.explain.s_nn,
                "s_nn_vec": it.explain.s_nn_vec,
                "s_nn_bm25": it.explain.s_nn_bm25,
                "s_jj": it.explain.s_jj,
                "coverage": it.explain.coverage,
                "margin_ratio": it.explain.margin_ratio,
                "p_nn": it.explain.p_nn,
                "p_jj": it.explain.p_jj,
                "matched_nn": it.explain.matched_nn,
                "matched_jj": it.explain.matched_jj,
            }
        return out

    rows = []
    if res.best is not None:
        rows.append(res.best)
    rows.extend(list(res.alternatives))
    rows = rows[:top_n]
    return JSONResponse({"items": [_item_out(r) for r in rows if r is not None]})
