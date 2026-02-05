from __future__ import annotations

import hashlib

import numpy as np

from item_search import Item, ItemSearchEngine, SearchRequest
from item_search.models import SearchConfig, Thresholds


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


def test_head_noun_is_counted_as_jj_when_has_jj() -> None:
    engine = ItemSearchEngine(FakeModel())
    engine.load_items(
        [
            Item(id="car-1", name="卡车", desc_labels=("破损", "蓝色", "重型")),
            Item(id="car-2", name="卡车", desc_labels=("破损", "蓝色", "轻型")),
        ],
        enable_bm25=False,
    )

    cfg = SearchConfig(top_k=2, enable_bm25=False, thresholds=Thresholds(tau_cover=0.99))
    req = SearchRequest(
        query="破损 蓝色 卡车",
        config=cfg,
        debug=True,
        candidate_ids=("car-1", "car-2"),
        pos_backend="jieba",
    )
    res = engine.search(req)
    assert res.parsed.head_noun == "卡车"
    assert "卡车" not in res.parsed.jj

    row = res.best or (res.alternatives[0] if res.alternatives else None)
    assert row is not None and row.explain is not None
    matched_terms = [m.get("jj") for m in row.explain.matched_jj]
    assert "卡车" in matched_terms
    assert len(row.explain.matched_jj) == len(res.parsed.jj) + 1

