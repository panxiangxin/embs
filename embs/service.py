from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from item_search import Item as EngineItem
from item_search import ItemSearchEngine

from .store import CatalogMeta


@dataclass(frozen=True)
class IndexInfo:
    catalog_version: int
    built_at_ms: int
    build_ms: float
    item_count: int
    name_phrase_count: int
    desc_label_count: int
    neg_name_samples: int
    neg_desc_samples: int
    enable_bm25: bool


class ItemSearchService:
    def __init__(self, *, engine: ItemSearchEngine, store, enable_bm25_default: bool = True) -> None:
        self._engine = engine
        self._store = store
        self._enable_bm25_default = bool(enable_bm25_default)
        self._lock = threading.RLock()
        self._index: IndexInfo | None = None

    @property
    def engine(self) -> ItemSearchEngine:
        return self._engine

    def index_info(self) -> IndexInfo | None:
        return self._index

    def current_catalog_meta(self) -> CatalogMeta:
        return self._store.get_meta()

    def rebuild_index(self, *, enable_bm25: bool | None = None) -> IndexInfo:
        enable_bm25 = self._enable_bm25_default if enable_bm25 is None else bool(enable_bm25)
        start = time.time()
        built_at_ms = int(time.time_ns() // 1_000_000)
        with self._lock:
            items = self._store.export_items(status="active")  # type: ignore[arg-type]
            engine_items = [
                EngineItem(
                    id=it.item_id,
                    name=it.name,
                    aliases=it.aliases,
                    type=it.type,  # type: ignore[arg-type]
                    desc_labels=it.desc_labels,
                )
                for it in items
            ]
            stats = self._engine.load_items(engine_items, enable_bm25=enable_bm25)
            meta = self._store.get_meta()

            info = IndexInfo(
                catalog_version=int(meta.catalog_version),
                built_at_ms=built_at_ms,
                build_ms=float((time.time() - start) * 1000.0),
                item_count=int(stats.item_count),
                name_phrase_count=int(stats.name_phrase_count),
                desc_label_count=int(stats.desc_label_count),
                neg_name_samples=int(stats.neg_name_samples),
                neg_desc_samples=int(stats.neg_desc_samples),
                enable_bm25=enable_bm25,
            )
            self._index = info
            return info

    def refresh_index(self) -> IndexInfo | None:
        meta = self._store.get_meta()
        cur = self._index
        if cur is not None and int(meta.catalog_version) == int(cur.catalog_version):
            return None
        return self.rebuild_index(enable_bm25=(cur.enable_bm25 if cur is not None else None))
