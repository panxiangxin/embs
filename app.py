from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from embs.settings import Settings
from embs.service import IndexInfo, ItemSearchService
from embs.store import FileItemStore, ItemAlreadyExistsError, ItemNotFoundError, SqliteItemStore, StoreError
from item_search import SearchConfig, SearchRequest as EngineSearchRequest
from item_search.models import Heuristics, Thresholds, Weights


def _get_store(settings: Settings):
    if settings.storage_backend == "sqlite":
        return SqliteItemStore(settings.sqlite_path)
    return FileItemStore(settings.items_path)


def _require_admin(request: Request) -> None:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    expected = settings.admin_api_key if settings is not None else None
    if not expected:
        return
    provided = request.headers.get("x-api-key")
    if not provided:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="unauthorized")


def _index_to_dict(info: IndexInfo) -> dict[str, Any]:
    return {
        "catalog_version": int(info.catalog_version),
        "built_at_ms": int(info.built_at_ms),
        "build_ms": float(info.build_ms),
        "item_count": int(info.item_count),
        "name_phrase_count": int(info.name_phrase_count),
        "desc_label_count": int(info.desc_label_count),
        "neg_name_samples": int(info.neg_name_samples),
        "neg_desc_samples": int(info.neg_desc_samples),
        "enable_bm25": bool(info.enable_bm25),
    }


def _item_record_to_dict(it: Any) -> dict[str, Any]:
    return {
        "id": it.item_id,
        "item_id": it.item_id,
        "name": it.name,
        "type": it.type,
        "aliases": list(it.aliases),
        "desc_labels": list(it.desc_labels),
        "labels": list(it.desc_labels),
        "attrs": dict(it.attrs or {}),
        "status": it.status,
        "version": int(it.version),
        "updated_at_ms": int(it.updated_at_ms),
    }


class ItemIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    item_id: str = Field(alias="id")
    name: str
    aliases: list[str] = Field(default_factory=list)
    type: str | list[str] | None = None
    desc_labels: list[str] = Field(default_factory=list, alias="labels")
    attrs: dict[str, Any] = Field(default_factory=dict)
    status: str | None = None


class ItemUpdateIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str
    aliases: list[str] = Field(default_factory=list)
    type: str | list[str] | None = None
    desc_labels: list[str] = Field(default_factory=list, alias="labels")
    attrs: dict[str, Any] = Field(default_factory=dict)
    status: str | None = None


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
    debug: bool = False


class ScoreTopNIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    query: str
    item_ids: list[str] = Field(default_factory=list, alias="candidate_ids")
    top_n: int = Field(default=10, ge=1)
    debug: bool = False
    config: SearchConfigIn = Field(default_factory=SearchConfigIn)
    pos_backend: str | None = None


class IndexRebuildIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enable_bm25: bool | None = None


def create_app(
    *,
    settings: Settings | None = None,
    model_override: object | None = None,
    store_override: object | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env(base_dir=Path(__file__).resolve().parent)
    app = FastAPI(title="embs: Item Catalog + Semantic Search API", version="2.0.0")

    app.state.settings = settings
    app.state.model = model_override
    app.state.store = store_override
    app.state.service = None

    demo_dir = Path(__file__).resolve().parent / "demo"
    if demo_dir.exists():
        app.mount("/demo", StaticFiles(directory=str(demo_dir), html=True), name="demo")

        @app.get("/")
        def demo_index() -> RedirectResponse:
            return RedirectResponse(url="/demo/")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup() -> None:
        if app.state.store is None:
            app.state.store = _get_store(settings)

        if app.state.model is None:
            from sentence_transformers import SentenceTransformer

            app.state.model = SentenceTransformer(settings.model_name, device=settings.device)

        from item_search import ItemSearchEngine

        engine = ItemSearchEngine(app.state.model)
        app.state.service = ItemSearchService(engine=engine, store=app.state.store, enable_bm25_default=settings.enable_bm25_default)

        if settings.auto_rebuild_on_startup:
            _rows, total = app.state.store.list_items(limit=1, status="active")  # type: ignore[attr-defined]
            if int(total) > 0:
                app.state.service.rebuild_index()

    @app.get("/health")
    def health() -> dict[str, Any]:
        svc: ItemSearchService | None = getattr(app.state, "service", None)
        store = getattr(app.state, "store", None)
        meta = store.get_meta() if store is not None else None
        idx = svc.index_info() if svc is not None else None
        return {
            "status": "ok" if svc is not None else "starting",
            "model": settings.model_name,
            "device": settings.device,
            "storage_backend": settings.storage_backend,
            "catalog_version": int(meta.catalog_version) if meta is not None else 0,
            "items_loaded": int(svc.engine.loaded_item_count()) if svc is not None else 0,
            "index_catalog_version": int(idx.catalog_version) if idx is not None else 0,
        }

    @app.get("/v1/status")
    def status() -> JSONResponse:
        store = getattr(app.state, "store", None)
        svc: ItemSearchService | None = getattr(app.state, "service", None)
        meta = store.get_meta() if store is not None else None
        idx = svc.index_info() if svc is not None else None
        return JSONResponse(
            {
                "model": settings.model_name,
                "device": settings.device,
                "storage_backend": settings.storage_backend,
                "data_dir": str(settings.data_dir),
                "catalog": {
                    "catalog_version": int(meta.catalog_version) if meta is not None else 0,
                    "updated_at_ms": int(meta.updated_at_ms) if meta is not None else 0,
                },
                "index": None if idx is None else _index_to_dict(idx),
            }
        )

    @app.options("/v1/embeddings")
    def embeddings_options() -> Response:
        return Response(status_code=204)

    @app.post("/v1/embeddings")
    def embeddings(payload: dict[str, Any]) -> JSONResponse:
        m = getattr(app.state, "model", None)
        if m is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if "model" in payload and payload.get("model") != settings.model_name:
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
        vectors = m.encode(texts, normalize_embeddings=normalize)
        data = [{"object": "embedding", "index": i, "embedding": emb.tolist()} for i, emb in enumerate(vectors)]
        return JSONResponse({"object": "list", "model": settings.model_name, "data": data, "usage": {"prompt_tokens": 0, "total_tokens": 0}})

    @app.get("/v1/items")
    def list_items(offset: int = 0, limit: int = 50, keyword: str | None = None, status: str | None = None) -> JSONResponse:
        store = getattr(app.state, "store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="Store not ready")
        status_norm = str(status).strip().lower() if status else None
        if status_norm and status_norm not in ("active", "disabled", "deleted"):
            raise HTTPException(status_code=400, detail="invalid status filter")
        rows, total = store.list_items(offset=offset, limit=limit, keyword=keyword, status=status_norm)
        return JSONResponse({"items": [_item_record_to_dict(r) for r in rows], "total": int(total), "offset": int(offset), "limit": int(limit)})

    @app.get("/v1/items/{item_id}")
    def get_item(item_id: str) -> JSONResponse:
        store = getattr(app.state, "store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="Store not ready")
        rec = store.get_item(str(item_id))
        if rec is None:
            raise HTTPException(status_code=404, detail="item not found")
        return JSONResponse(_item_record_to_dict(rec))

    @app.post("/v1/items")
    def create_item(payload: ItemIn, _: None = Depends(_require_admin)) -> JSONResponse:
        store = getattr(app.state, "store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="Store not ready")
        status_norm = (payload.status or "active").strip().lower()
        if status_norm not in ("active", "disabled"):
            raise HTTPException(status_code=400, detail="invalid status")
        try:
            rec = store.create_item(
                item_id=payload.item_id,
                name=payload.name,
                type=payload.type,
                aliases=payload.aliases,
                desc_labels=payload.desc_labels,
                attrs=payload.attrs,
                status=status_norm,
            )
        except ItemAlreadyExistsError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except StoreError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return JSONResponse(_item_record_to_dict(rec))

    @app.put("/v1/items/{item_id}")
    def update_item(item_id: str, payload: ItemUpdateIn, _: None = Depends(_require_admin)) -> JSONResponse:
        store = getattr(app.state, "store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="Store not ready")
        status_norm = (payload.status or "").strip().lower() or None
        if status_norm is not None and status_norm not in ("active", "disabled"):
            raise HTTPException(status_code=400, detail="invalid status")
        try:
            rec = store.update_item(
                item_id=item_id,
                name=payload.name,
                type=payload.type,
                aliases=payload.aliases,
                desc_labels=payload.desc_labels,
                attrs=payload.attrs,
                status=status_norm,
            )
        except ItemNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except StoreError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return JSONResponse(_item_record_to_dict(rec))

    @app.delete("/v1/items/{item_id}")
    def delete_item(item_id: str, mode: str = "deleted", _: None = Depends(_require_admin)) -> JSONResponse:
        store = getattr(app.state, "store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="Store not ready")
        mode_norm = str(mode).strip().lower()
        if mode_norm not in ("deleted", "disabled"):
            raise HTTPException(status_code=400, detail="mode must be deleted|disabled")
        try:
            rec = store.set_status(item_id=item_id, status=mode_norm)
        except ItemNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except StoreError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return JSONResponse(_item_record_to_dict(rec))

    @app.post("/v1/items/import")
    async def import_items(request: Request, rebuild: bool = False, mode: str = "upsert", _: None = Depends(_require_admin)) -> JSONResponse:
        store = getattr(app.state, "store", None)
        svc: ItemSearchService | None = getattr(app.state, "service", None)
        if store is None or svc is None:
            raise HTTPException(status_code=503, detail="Service not ready")

        ct = (request.headers.get("content-type") or "").lower()
        raw_body = await request.body()
        if not raw_body:
            raise HTTPException(status_code=400, detail="empty body")

        payload: Any
        if "application/x-ndjson" in ct or "application/ndjson" in ct or "text/plain" in ct:
            items = []
            for line in raw_body.decode("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
            payload = {"items": items}
        else:
            payload = json.loads(raw_body.decode("utf-8"))

        enable_bm25 = True
        items_raw: list[Any]
        if isinstance(payload, dict):
            enable_bm25 = bool(payload.get("enable_bm25", True))
            items_raw = payload.get("items") or []
        elif isinstance(payload, list):
            items_raw = payload
        else:
            raise HTTPException(status_code=400, detail="invalid import payload")

        if not isinstance(items_raw, list):
            raise HTTPException(status_code=400, detail="items must be a list")

        mode_norm = str(mode).strip().lower()
        if mode_norm not in ("upsert", "replace"):
            raise HTTPException(status_code=400, detail="mode must be upsert|replace")

        try:
            res = store.replace_all(items_raw) if mode_norm == "replace" else store.bulk_upsert(items_raw)
        except (StoreError, ValueError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        index = svc.rebuild_index(enable_bm25=enable_bm25) if rebuild else None
        return JSONResponse({"ok": True, "result": res, "rebuild": bool(rebuild), "index": None if index is None else _index_to_dict(index)})

    @app.get("/v1/items/export")
    def export_items(status: str | None = None, format: str = "json") -> Response:
        store = getattr(app.state, "store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="Store not ready")
        status_norm = str(status).strip().lower() if status else None
        if status_norm and status_norm not in ("active", "disabled", "deleted"):
            raise HTTPException(status_code=400, detail="invalid status filter")
        rows = store.export_items(status=status_norm)
        meta = store.get_meta()
        format_norm = str(format or "json").strip().lower()
        if format_norm in ("ndjson", "application/x-ndjson"):
            lines = [json.dumps(_item_record_to_dict(r), ensure_ascii=False) for r in rows]
            body = "\n".join(lines) + ("\n" if lines else "")
            return Response(content=body, media_type="application/x-ndjson")
        return JSONResponse(
            {
                "meta": {"catalog_version": int(meta.catalog_version), "updated_at_ms": int(meta.updated_at_ms)},
                "items": [_item_record_to_dict(r) for r in rows],
            }
        )

    @app.get("/v1/index/info")
    def index_info() -> JSONResponse:
        store = getattr(app.state, "store", None)
        svc: ItemSearchService | None = getattr(app.state, "service", None)
        if store is None or svc is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        meta = store.get_meta()
        idx = svc.index_info()
        return JSONResponse({"catalog": {"catalog_version": int(meta.catalog_version), "updated_at_ms": int(meta.updated_at_ms)}, "index": None if idx is None else _index_to_dict(idx)})

    @app.post("/v1/index/rebuild")
    def index_rebuild(payload: IndexRebuildIn, _: None = Depends(_require_admin)) -> JSONResponse:
        svc: ItemSearchService | None = getattr(app.state, "service", None)
        if svc is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        info = svc.rebuild_index(enable_bm25=payload.enable_bm25)
        return JSONResponse({"ok": True, "index": _index_to_dict(info)})

    @app.post("/v1/index/refresh")
    def index_refresh(_: None = Depends(_require_admin)) -> JSONResponse:
        svc: ItemSearchService | None = getattr(app.state, "service", None)
        if svc is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        info = svc.refresh_index()
        return JSONResponse({"ok": True, "changed": info is not None, "index": None if info is None else _index_to_dict(info)})

    @app.post("/v1/items/load")
    def load_items_compat(payload: LoadItemsIn, _: None = Depends(_require_admin)) -> JSONResponse:
        """Compatibility endpoint for the original demo: replace catalog and rebuild index."""
        store = getattr(app.state, "store", None)
        svc: ItemSearchService | None = getattr(app.state, "service", None)
        if store is None or svc is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        res = store.replace_all([i.model_dump(by_alias=True) for i in payload.items])
        info = svc.rebuild_index(enable_bm25=payload.enable_bm25)
        return JSONResponse({"ok": True, "result": res, "index": _index_to_dict(info)})

    @app.post("/v1/item_search")
    def item_search(payload: ItemSearchIn) -> JSONResponse:
        svc: ItemSearchService | None = getattr(app.state, "service", None)
        if svc is None:
            raise HTTPException(status_code=503, detail="Search service not ready")
        if svc.engine.loaded_item_count() <= 0:
            raise HTTPException(status_code=503, detail="No items loaded. Call /v1/index/rebuild or /v1/items/import?rebuild=true.")

        candidate_ids = tuple(payload.candidate_ids) if payload.candidate_ids is not None else None
        top_k = len(candidate_ids) if candidate_ids else 5
        cfg = SearchConfig(top_k=top_k, enable_bm25=settings.enable_bm25_default)
        req = EngineSearchRequest(
            query=payload.query,
            config=cfg,
            debug=False,
            candidate_ids=candidate_ids,
            pos_backend=settings.pos_backend_default,
        )

        try:
            res = svc.engine.search(req)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        rows = []
        if res.best is not None:
            rows.append(res.best)
        rows.extend(list(res.alternatives))

        return JSONResponse(
            {
                "candidates": [{"id": it.id, "score": it.score} for it in rows if it is not None],
                "debug": {"parsed": {"nn": list(res.parsed.nn), "jj": list(res.parsed.jj), "head_noun": res.parsed.head_noun}},
            }
        )

    @app.post("/v1/item_search/topn")
    def item_search_topn(payload: ScoreTopNIn) -> JSONResponse:
        svc: ItemSearchService | None = getattr(app.state, "service", None)
        if svc is None:
            raise HTTPException(status_code=503, detail="Search service not ready")
        if svc.engine.loaded_item_count() <= 0:
            raise HTTPException(status_code=503, detail="No items loaded. Call /v1/index/rebuild first.")

        pos_backend = (payload.pos_backend or settings.pos_backend_default).strip().lower()
        if pos_backend not in ("jieba", "hanlp"):
            raise HTTPException(status_code=400, detail="pos_backend must be jieba|hanlp")

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
            pos_backend=pos_backend,
        )

        try:
            res = svc.engine.search(req)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        def _scored_out(it):
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
        return JSONResponse({"items": [_scored_out(r) for r in rows if r is not None]})

    return app


app = create_app()
