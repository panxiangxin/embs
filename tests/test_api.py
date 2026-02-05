from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import create_app  # noqa: E402
from embs.settings import Settings
from embs.store import FileItemStore, SqliteItemStore


class FakeModel:
    def __init__(self, dim: int = 16) -> None:
        self._dim = int(dim)

    def encode(self, texts, normalize_embeddings: bool = True):
        rows = []
        for t in texts:
            b = hashlib.md5(str(t).encode("utf-8")).digest()
            v = np.frombuffer(b, dtype=np.uint8).astype(np.float32)
            if self._dim > v.shape[0]:
                reps = int(np.ceil(self._dim / v.shape[0]))
                v = np.tile(v, reps)[: self._dim]
            else:
                v = v[: self._dim]
            v = v - float(np.mean(v))
            if normalize_embeddings:
                n = float(np.linalg.norm(v))
                if n > 0:
                    v = v / n
            rows.append(v)
        return np.stack(rows, axis=0).astype(np.float32, copy=False)


SAMPLE_PAYLOAD = {
    "enable_bm25": True,
    "items": [
        {
            "id": "veh-01",
            "name": "蓝色卡车",
            "type": ["卡车", "车", "汽车", "货车"],
            "aliases": ["truck"],
            "desc_labels": ["蓝色", "破损", "重型"],
        },
        {
            "id": "veh-02",
            "name": "红色皮卡",
            "type": ["皮卡", "车", "汽车"],
            "desc_labels": ["红色", "完好"],
        },
        {
            "id": "key-01",
            "name": "铁钥匙",
            "type": ["钥匙", "金属钥匙"],
            "desc_labels": ["金属", "旧", "小"],
        },
    ],
}


def _make_settings(tmp_path: Path, backend: str) -> Settings:
    data_dir = tmp_path / "data"
    return Settings(
        model_name="fake",
        device="cpu",
        storage_backend=backend,  # type: ignore[arg-type]
        data_dir=data_dir,
        items_path=data_dir / "items.json",
        sqlite_path=data_dir / "embs.db",
        admin_api_key="secret",
        pos_backend_default="jieba",
        enable_bm25_default=True,
        auto_rebuild_on_startup=False,
    )


@pytest.fixture(params=["file", "sqlite"])
def client(tmp_path: Path, request: pytest.FixtureRequest):
    backend = str(request.param)
    settings = _make_settings(tmp_path, backend)
    store = FileItemStore(settings.items_path) if backend == "file" else SqliteItemStore(settings.sqlite_path)
    app = create_app(settings=settings, model_override=FakeModel(), store_override=store)
    with TestClient(app) as c:
        yield c


def test_admin_key_required(client: TestClient) -> None:
    r = client.post("/v1/items", json={"id": "itm-01", "name": "木箱"})
    assert r.status_code == 401

    headers = {"x-api-key": "secret"}
    r = client.post("/v1/items", headers=headers, json={"id": "itm-01", "name": "木箱", "type": ["箱子"]})
    assert r.status_code == 200
    item = r.json()
    assert item["id"] == "itm-01"
    assert item["version"] == 1

    r = client.put("/v1/items/itm-01", headers=headers, json={"name": "大木箱", "desc_labels": ["木制", "大"]})
    assert r.status_code == 200
    item2 = r.json()
    assert item2["id"] == "itm-01"
    assert item2["version"] == 2

    r = client.get("/v1/items")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 1


def test_import_rebuild_and_search(client: TestClient) -> None:
    headers = {"x-api-key": "secret"}
    r = client.post("/v1/items/import?rebuild=true&mode=replace", headers=headers, json=SAMPLE_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["index"]["item_count"] == 3

    r = client.post("/v1/item_search", json={"query": "破损的蓝色卡车", "debug": True})
    assert r.status_code == 200
    out = r.json()
    assert isinstance(out.get("candidates"), list)
    assert out["candidates"]
    assert "debug" in out and "parsed" in out["debug"]
    assert set(out["debug"]["parsed"].keys()) == {"nn", "jj", "head_noun"}
    assert "id" in out["candidates"][0] and "score" in out["candidates"][0]

    cand = ["veh-02"]
    r = client.post("/v1/item_search", json={"query": "车", "candidate_ids": cand})
    assert r.status_code == 200
    out = r.json()
    ids = [c["id"] for c in out.get("candidates") or []]
    assert ids == cand


def test_index_refresh(client: TestClient) -> None:
    headers = {"x-api-key": "secret"}
    r = client.post("/v1/items/import?rebuild=true&mode=replace", headers=headers, json=SAMPLE_PAYLOAD)
    assert r.status_code == 200

    r = client.get("/v1/index/info")
    assert r.status_code == 200
    idx0 = r.json()["index"]["catalog_version"]

    r = client.post("/v1/items", headers=headers, json={"id": "box-99", "name": "神秘箱子", "type": ["箱子"]})
    assert r.status_code == 200

    r = client.post("/v1/index/refresh", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["changed"] is True
    assert body["index"]["catalog_version"] > idx0

    r = client.post("/v1/index/refresh", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["changed"] is False
    assert body["index"] is None
