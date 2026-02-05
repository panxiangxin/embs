from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

StorageBackend = Literal["file", "sqlite"]


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


@dataclass(frozen=True)
class Settings:
    model_name: str
    device: str
    storage_backend: StorageBackend
    data_dir: Path
    items_path: Path
    sqlite_path: Path
    admin_api_key: str | None
    pos_backend_default: Literal["jieba", "hanlp"]
    enable_bm25_default: bool
    auto_rebuild_on_startup: bool

    @staticmethod
    def from_env(base_dir: Path | None = None) -> "Settings":
        base_dir = base_dir or Path(__file__).resolve().parents[1]

        model_name = os.getenv("MODEL_NAME", "BAAI/bge-small-zh-v1.5")
        device = os.getenv("DEVICE", "cpu")

        storage_backend = str(os.getenv("STORAGE_BACKEND", "file")).strip().lower()
        if storage_backend not in ("file", "sqlite"):
            storage_backend = "file"

        data_dir_raw = os.getenv("DATA_DIR", "data")
        data_dir = Path(data_dir_raw)
        if not data_dir.is_absolute():
            data_dir = base_dir / data_dir

        items_path_raw = os.getenv("ITEMS_PATH")
        items_path = Path(items_path_raw) if items_path_raw else (data_dir / "items.json")
        if not items_path.is_absolute():
            items_path = base_dir / items_path

        sqlite_path_raw = os.getenv("SQLITE_PATH")
        sqlite_path = Path(sqlite_path_raw) if sqlite_path_raw else (data_dir / "embs.db")
        if not sqlite_path.is_absolute():
            sqlite_path = base_dir / sqlite_path

        admin_api_key = os.getenv("ADMIN_API_KEY") or None

        pos_backend_default = str(os.getenv("POS_BACKEND_DEFAULT", "jieba")).strip().lower()
        if pos_backend_default not in ("jieba", "hanlp"):
            pos_backend_default = "jieba"

        enable_bm25_default = _parse_bool(os.getenv("ENABLE_BM25_DEFAULT"), default=True)
        auto_rebuild_on_startup = _parse_bool(os.getenv("AUTO_REBUILD_ON_STARTUP"), default=True)

        return Settings(
            model_name=model_name,
            device=device,
            storage_backend=storage_backend,  # type: ignore[arg-type]
            data_dir=data_dir,
            items_path=items_path,
            sqlite_path=sqlite_path,
            admin_api_key=admin_api_key,
            pos_backend_default=pos_backend_default,  # type: ignore[arg-type]
            enable_bm25_default=enable_bm25_default,
            auto_rebuild_on_startup=auto_rebuild_on_startup,
        )

