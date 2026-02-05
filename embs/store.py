from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, cast

ItemStatus = Literal["active", "disabled", "deleted"]


def _now_ms() -> int:
    return int(time.time_ns() // 1_000_000)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CatalogMeta:
    catalog_version: int
    updated_at_ms: int


@dataclass(frozen=True)
class ItemRecord:
    item_id: str
    name: str
    type: str | list[str] | None = None
    aliases: tuple[str, ...] = ()
    desc_labels: tuple[str, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)
    status: ItemStatus = "active"
    version: int = 1
    updated_at_ms: int = 0


class StoreError(RuntimeError):
    pass


class ItemAlreadyExistsError(StoreError):
    pass


class ItemNotFoundError(StoreError):
    pass


class FileItemStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        _ensure_parent_dir(self._path)

    def get_meta(self) -> CatalogMeta:
        with self._lock:
            meta, _items = self._read()
            return meta

    def get_item(self, item_id: str) -> ItemRecord | None:
        with self._lock:
            _meta, items = self._read()
            return items.get(str(item_id))

    def list_items(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        keyword: str | None = None,
        status: ItemStatus | None = None,
    ) -> tuple[list[ItemRecord], int]:
        keyword_norm = (keyword or "").strip().lower()
        with self._lock:
            _meta, items = self._read()
            rows = list(items.values())

        if status is not None:
            rows = [r for r in rows if r.status == status]
        if keyword_norm:
            rows = [r for r in rows if keyword_norm in r.item_id.lower() or keyword_norm in r.name.lower()]

        rows.sort(key=lambda r: (r.status != "active", -r.updated_at_ms, r.item_id))
        total = len(rows)
        offset = max(0, int(offset))
        limit = max(1, int(limit))
        return rows[offset : offset + limit], total

    def export_items(self, *, status: ItemStatus | None = None) -> list[ItemRecord]:
        with self._lock:
            _meta, items = self._read()
            rows = list(items.values())
        if status is not None:
            rows = [r for r in rows if r.status == status]
        rows.sort(key=lambda r: r.item_id)
        return rows

    def create_item(
        self,
        *,
        item_id: str,
        name: str,
        type: str | list[str] | None = None,
        aliases: Iterable[str] = (),
        desc_labels: Iterable[str] = (),
        attrs: dict[str, Any] | None = None,
        status: ItemStatus = "active",
    ) -> ItemRecord:
        if status == "deleted":
            raise StoreError("cannot create item with status=deleted")

        now_ms = _now_ms()
        with self._lock:
            meta, items = self._read()
            key = str(item_id)
            if key in items:
                raise ItemAlreadyExistsError(f"item already exists: {key}")
            rec = ItemRecord(
                item_id=key,
                name=str(name),
                type=type,
                aliases=tuple(str(a) for a in aliases if str(a).strip()),
                desc_labels=tuple(str(x) for x in desc_labels if str(x).strip()),
                attrs=dict(attrs or {}),
                status=status,
                version=1,
                updated_at_ms=now_ms,
            )
            items[key] = rec
            meta = CatalogMeta(catalog_version=int(meta.catalog_version) + 1, updated_at_ms=now_ms)
            self._write(meta, items)
            return rec

    def update_item(
        self,
        *,
        item_id: str,
        name: str,
        type: str | list[str] | None = None,
        aliases: Iterable[str] = (),
        desc_labels: Iterable[str] = (),
        attrs: dict[str, Any] | None = None,
        status: ItemStatus | None = None,
    ) -> ItemRecord:
        now_ms = _now_ms()
        with self._lock:
            meta, items = self._read()
            key = str(item_id)
            old = items.get(key)
            if old is None:
                raise ItemNotFoundError(f"item not found: {key}")

            new_status = old.status if status is None else status
            if new_status == "deleted":
                # allow update only via delete/disable API to keep semantics clear
                raise StoreError("use delete endpoint to set status=deleted")

            rec = ItemRecord(
                item_id=key,
                name=str(name),
                type=type,
                aliases=tuple(str(a) for a in aliases if str(a).strip()),
                desc_labels=tuple(str(x) for x in desc_labels if str(x).strip()),
                attrs=dict(attrs or {}),
                status=new_status,
                version=int(old.version) + 1,
                updated_at_ms=now_ms,
            )
            items[key] = rec
            meta = CatalogMeta(catalog_version=int(meta.catalog_version) + 1, updated_at_ms=now_ms)
            self._write(meta, items)
            return rec

    def set_status(self, *, item_id: str, status: ItemStatus) -> ItemRecord:
        now_ms = _now_ms()
        with self._lock:
            meta, items = self._read()
            key = str(item_id)
            old = items.get(key)
            if old is None:
                raise ItemNotFoundError(f"item not found: {key}")
            rec = ItemRecord(
                item_id=old.item_id,
                name=old.name,
                type=old.type,
                aliases=old.aliases,
                desc_labels=old.desc_labels,
                attrs=dict(old.attrs or {}),
                status=status,
                version=int(old.version) + 1,
                updated_at_ms=now_ms,
            )
            items[key] = rec
            meta = CatalogMeta(catalog_version=int(meta.catalog_version) + 1, updated_at_ms=now_ms)
            self._write(meta, items)
            return rec

    def bulk_upsert(self, items_in: Iterable[dict[str, Any]]) -> dict[str, int]:
        now_ms = _now_ms()
        created = 0
        updated = 0
        with self._lock:
            meta, items = self._read()
            for raw in items_in:
                rec_in = _parse_item_input(raw)
                old = items.get(rec_in.item_id)
                if old is None:
                    created += 1
                    items[rec_in.item_id] = ItemRecord(
                        item_id=rec_in.item_id,
                        name=rec_in.name,
                        type=rec_in.type,
                        aliases=rec_in.aliases,
                        desc_labels=rec_in.desc_labels,
                        attrs=dict(rec_in.attrs or {}),
                        status=rec_in.status,
                        version=1,
                        updated_at_ms=now_ms,
                    )
                else:
                    updated += 1
                    items[rec_in.item_id] = ItemRecord(
                        item_id=rec_in.item_id,
                        name=rec_in.name,
                        type=rec_in.type,
                        aliases=rec_in.aliases,
                        desc_labels=rec_in.desc_labels,
                        attrs=dict(rec_in.attrs or {}),
                        status=rec_in.status,
                        version=int(old.version) + 1,
                        updated_at_ms=now_ms,
                    )
            if created or updated:
                meta = CatalogMeta(catalog_version=int(meta.catalog_version) + 1, updated_at_ms=now_ms)
                self._write(meta, items)
        return {"created": created, "updated": updated}

    def replace_all(self, items_in: Iterable[dict[str, Any]]) -> dict[str, int]:
        now_ms = _now_ms()
        items: dict[str, ItemRecord] = {}
        count = 0
        for raw in items_in:
            rec_in = _parse_item_input(raw)
            items[rec_in.item_id] = ItemRecord(
                item_id=rec_in.item_id,
                name=rec_in.name,
                type=rec_in.type,
                aliases=rec_in.aliases,
                desc_labels=rec_in.desc_labels,
                attrs=dict(rec_in.attrs or {}),
                status=rec_in.status,
                version=1,
                updated_at_ms=now_ms,
            )
            count += 1
        with self._lock:
            meta = CatalogMeta(catalog_version=self._read()[0].catalog_version + 1, updated_at_ms=now_ms)
            self._write(meta, items)
        return {"replaced": count}

    def _read(self) -> tuple[CatalogMeta, dict[str, ItemRecord]]:
        if not self._path.exists():
            return CatalogMeta(catalog_version=0, updated_at_ms=0), {}

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise StoreError(f"bad items file: {self._path}") from e

        if isinstance(raw, list):
            meta_raw = {}
            items_raw = raw
        else:
            meta_raw = cast(dict[str, Any], raw.get("meta") or {})
            items_raw = raw.get("items") or []

        meta = CatalogMeta(
            catalog_version=int(meta_raw.get("catalog_version") or 0),
            updated_at_ms=int(meta_raw.get("updated_at_ms") or meta_raw.get("updated_at") or 0),
        )

        items: dict[str, ItemRecord] = {}
        for it in items_raw:
            if not isinstance(it, dict):
                continue
            rec = _parse_item_record(it)
            items[rec.item_id] = rec

        return meta, items

    def _write(self, meta: CatalogMeta, items: dict[str, ItemRecord]) -> None:
        payload = {
            "meta": {"catalog_version": int(meta.catalog_version), "updated_at_ms": int(meta.updated_at_ms)},
            "items": [_item_record_to_dict(v) for v in items.values()],
        }
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self._path)


class SqliteItemStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        _ensure_parent_dir(self._path)
        self._init_db()

    def get_meta(self) -> CatalogMeta:
        with self._connect() as conn:
            return _read_meta(conn)

    def get_item(self, item_id: str) -> ItemRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM items WHERE item_id = ?", (str(item_id),)).fetchone()
            return _row_to_item(row) if row else None

    def list_items(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        keyword: str | None = None,
        status: ItemStatus | None = None,
    ) -> tuple[list[ItemRecord], int]:
        offset = max(0, int(offset))
        limit = max(1, int(limit))
        keyword_norm = (keyword or "").strip()

        where = []
        params: list[Any] = []
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if keyword_norm:
            where.append("(item_id LIKE ? OR name LIKE ?)")
            kw = f"%{keyword_norm}%"
            params.extend([kw, kw])

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        with self._connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) AS c FROM items {where_sql}", params).fetchone()["c"])
            rows = conn.execute(
                f"SELECT * FROM items {where_sql} ORDER BY (status!='active'), updated_at_ms DESC, item_id ASC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            return [_row_to_item(r) for r in rows], total

    def export_items(self, *, status: ItemStatus | None = None) -> list[ItemRecord]:
        where_sql = ""
        params: list[Any] = []
        if status is not None:
            where_sql = "WHERE status = ?"
            params.append(status)
        with self._connect() as conn:
            rows = conn.execute(f"SELECT * FROM items {where_sql} ORDER BY item_id ASC", params).fetchall()
            return [_row_to_item(r) for r in rows]

    def create_item(
        self,
        *,
        item_id: str,
        name: str,
        type: str | list[str] | None = None,
        aliases: Iterable[str] = (),
        desc_labels: Iterable[str] = (),
        attrs: dict[str, Any] | None = None,
        status: ItemStatus = "active",
    ) -> ItemRecord:
        if status == "deleted":
            raise StoreError("cannot create item with status=deleted")
        now_ms = _now_ms()
        with self._lock, self._connect() as conn:
            try:
                conn.execute("BEGIN")
                conn.execute(
                    """
                    INSERT INTO items(item_id, name, type_json, aliases_json, desc_labels_json, attrs_json, status, version, updated_at_ms)
                    VALUES(?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        str(item_id),
                        str(name),
                        json.dumps(type, ensure_ascii=False),
                        json.dumps(list(aliases), ensure_ascii=False),
                        json.dumps(list(desc_labels), ensure_ascii=False),
                        json.dumps(attrs or {}, ensure_ascii=False),
                        status,
                        now_ms,
                    ),
                )
                _bump_meta(conn, now_ms)
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as e:
                conn.execute("ROLLBACK")
                raise ItemAlreadyExistsError(f"item already exists: {item_id}") from e

        rec = self.get_item(str(item_id))
        if rec is None:
            raise StoreError("create failed")
        return rec

    def update_item(
        self,
        *,
        item_id: str,
        name: str,
        type: str | list[str] | None = None,
        aliases: Iterable[str] = (),
        desc_labels: Iterable[str] = (),
        attrs: dict[str, Any] | None = None,
        status: ItemStatus | None = None,
    ) -> ItemRecord:
        now_ms = _now_ms()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN")
            old = conn.execute("SELECT status FROM items WHERE item_id = ?", (str(item_id),)).fetchone()
            if old is None:
                conn.execute("ROLLBACK")
                raise ItemNotFoundError(f"item not found: {item_id}")

            new_status = old["status"] if status is None else status
            if new_status == "deleted":
                conn.execute("ROLLBACK")
                raise StoreError("use delete endpoint to set status=deleted")

            conn.execute(
                """
                UPDATE items
                SET name = ?,
                    type_json = ?,
                    aliases_json = ?,
                    desc_labels_json = ?,
                    attrs_json = ?,
                    status = ?,
                    version = version + 1,
                    updated_at_ms = ?
                WHERE item_id = ?
                """,
                (
                    str(name),
                    json.dumps(type, ensure_ascii=False),
                    json.dumps(list(aliases), ensure_ascii=False),
                    json.dumps(list(desc_labels), ensure_ascii=False),
                    json.dumps(attrs or {}, ensure_ascii=False),
                    str(new_status),
                    now_ms,
                    str(item_id),
                ),
            )
            _bump_meta(conn, now_ms)
            conn.execute("COMMIT")

        rec = self.get_item(str(item_id))
        if rec is None:
            raise StoreError("update failed")
        return rec

    def set_status(self, *, item_id: str, status: ItemStatus) -> ItemRecord:
        now_ms = _now_ms()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN")
            cur = conn.execute("SELECT item_id FROM items WHERE item_id = ?", (str(item_id),)).fetchone()
            if cur is None:
                conn.execute("ROLLBACK")
                raise ItemNotFoundError(f"item not found: {item_id}")
            conn.execute(
                "UPDATE items SET status = ?, version = version + 1, updated_at_ms = ? WHERE item_id = ?",
                (status, now_ms, str(item_id)),
            )
            _bump_meta(conn, now_ms)
            conn.execute("COMMIT")

        rec = self.get_item(str(item_id))
        if rec is None:
            raise StoreError("set_status failed")
        return rec

    def bulk_upsert(self, items_in: Iterable[dict[str, Any]]) -> dict[str, int]:
        now_ms = _now_ms()
        created = 0
        updated = 0
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN")
            for raw in items_in:
                rec_in = _parse_item_input(raw)
                exists = conn.execute("SELECT 1 FROM items WHERE item_id = ?", (rec_in.item_id,)).fetchone()
                if exists is None:
                    created += 1
                else:
                    updated += 1
                conn.execute(
                    """
                    INSERT INTO items(item_id, name, type_json, aliases_json, desc_labels_json, attrs_json, status, version, updated_at_ms)
                    VALUES(?, ?, ?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(item_id) DO UPDATE SET
                        name = excluded.name,
                        type_json = excluded.type_json,
                        aliases_json = excluded.aliases_json,
                        desc_labels_json = excluded.desc_labels_json,
                        attrs_json = excluded.attrs_json,
                        status = excluded.status,
                        version = items.version + 1,
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    (
                        rec_in.item_id,
                        rec_in.name,
                        json.dumps(rec_in.type, ensure_ascii=False),
                        json.dumps(list(rec_in.aliases), ensure_ascii=False),
                        json.dumps(list(rec_in.desc_labels), ensure_ascii=False),
                        json.dumps(rec_in.attrs or {}, ensure_ascii=False),
                        rec_in.status,
                        now_ms,
                    ),
                )
            if created or updated:
                _bump_meta(conn, now_ms)
            conn.execute("COMMIT")
        return {"created": created, "updated": updated}

    def replace_all(self, items_in: Iterable[dict[str, Any]]) -> dict[str, int]:
        now_ms = _now_ms()
        items = [_parse_item_input(x) for x in items_in]
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM items")
            for rec_in in items:
                conn.execute(
                    """
                    INSERT INTO items(item_id, name, type_json, aliases_json, desc_labels_json, attrs_json, status, version, updated_at_ms)
                    VALUES(?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        rec_in.item_id,
                        rec_in.name,
                        json.dumps(rec_in.type, ensure_ascii=False),
                        json.dumps(list(rec_in.aliases), ensure_ascii=False),
                        json.dumps(list(rec_in.desc_labels), ensure_ascii=False),
                        json.dumps(rec_in.attrs or {}, ensure_ascii=False),
                        rec_in.status,
                        now_ms,
                    ),
                )
            _bump_meta(conn, now_ms)
            conn.execute("COMMIT")
        return {"replaced": len(items)}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS items(
                    item_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type_json TEXT,
                    aliases_json TEXT,
                    desc_labels_json TEXT,
                    attrs_json TEXT,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            if conn.execute("SELECT 1 FROM meta WHERE key='catalog_version'").fetchone() is None:
                conn.execute("INSERT INTO meta(key, value) VALUES('catalog_version', '0')")
            if conn.execute("SELECT 1 FROM meta WHERE key='updated_at_ms'").fetchone() is None:
                conn.execute("INSERT INTO meta(key, value) VALUES('updated_at_ms', '0')")


@dataclass(frozen=True)
class _ItemInput:
    item_id: str
    name: str
    type: str | list[str] | None
    aliases: tuple[str, ...]
    desc_labels: tuple[str, ...]
    attrs: dict[str, Any]
    status: ItemStatus


def _parse_item_input(raw: dict[str, Any]) -> _ItemInput:
    item_id = raw.get("item_id")
    if item_id is None:
        item_id = raw.get("id")
    if item_id is None:
        raise StoreError("missing item_id/id")

    name = raw.get("name")
    if name is None:
        raise StoreError(f"missing name for item {item_id}")

    raw_type = raw.get("type")
    if isinstance(raw_type, list):
        type_value: str | list[str] | None = [str(x) for x in raw_type if str(x).strip()]
    elif raw_type is None:
        type_value = None
    else:
        type_value = str(raw_type)

    aliases_raw = raw.get("aliases") or []
    if not isinstance(aliases_raw, list):
        aliases_raw = [aliases_raw]
    aliases = tuple(str(x) for x in aliases_raw if str(x).strip())

    labels_raw = raw.get("desc_labels")
    if labels_raw is None:
        labels_raw = raw.get("labels")
    if labels_raw is None:
        labels_raw = []
    if not isinstance(labels_raw, list):
        labels_raw = [labels_raw]
    desc_labels = tuple(str(x) for x in labels_raw if str(x).strip())

    attrs_raw = raw.get("attrs")
    attrs = dict(attrs_raw) if isinstance(attrs_raw, dict) else {}

    status_raw = str(raw.get("status") or "active").strip().lower()
    if status_raw not in ("active", "disabled", "deleted"):
        status_raw = "active"

    return _ItemInput(
        item_id=str(item_id),
        name=str(name),
        type=type_value,
        aliases=aliases,
        desc_labels=desc_labels,
        attrs=attrs,
        status=cast(ItemStatus, status_raw),
    )


def _parse_item_record(raw: dict[str, Any]) -> ItemRecord:
    rec_in = _parse_item_input(raw)
    version = int(raw.get("version") or 1)
    updated_at_ms = int(raw.get("updated_at_ms") or raw.get("updated_at") or 0)
    return ItemRecord(
        item_id=rec_in.item_id,
        name=rec_in.name,
        type=rec_in.type,
        aliases=rec_in.aliases,
        desc_labels=rec_in.desc_labels,
        attrs=dict(rec_in.attrs or {}),
        status=rec_in.status,
        version=version,
        updated_at_ms=updated_at_ms,
    )


def _item_record_to_dict(rec: ItemRecord) -> dict[str, Any]:
    return {
        "item_id": rec.item_id,
        "name": rec.name,
        "type": rec.type,
        "aliases": list(rec.aliases),
        "desc_labels": list(rec.desc_labels),
        "attrs": dict(rec.attrs or {}),
        "status": rec.status,
        "version": int(rec.version),
        "updated_at_ms": int(rec.updated_at_ms),
    }


def _row_to_item(row: sqlite3.Row) -> ItemRecord:
    raw_type = json.loads(row["type_json"]) if row["type_json"] else None
    aliases = json.loads(row["aliases_json"]) if row["aliases_json"] else []
    labels = json.loads(row["desc_labels_json"]) if row["desc_labels_json"] else []
    attrs = json.loads(row["attrs_json"]) if row["attrs_json"] else {}
    return ItemRecord(
        item_id=str(row["item_id"]),
        name=str(row["name"]),
        type=raw_type,
        aliases=tuple(str(x) for x in aliases if str(x).strip()),
        desc_labels=tuple(str(x) for x in labels if str(x).strip()),
        attrs=dict(attrs) if isinstance(attrs, dict) else {},
        status=cast(ItemStatus, str(row["status"])),
        version=int(row["version"]),
        updated_at_ms=int(row["updated_at_ms"]),
    )


def _read_meta(conn: sqlite3.Connection) -> CatalogMeta:
    version = conn.execute("SELECT value FROM meta WHERE key='catalog_version'").fetchone()
    updated = conn.execute("SELECT value FROM meta WHERE key='updated_at_ms'").fetchone()
    return CatalogMeta(catalog_version=int(version["value"]) if version else 0, updated_at_ms=int(updated["value"]) if updated else 0)


def _bump_meta(conn: sqlite3.Connection, now_ms: int) -> None:
    conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='catalog_version'")
    conn.execute("UPDATE meta SET value = ? WHERE key='updated_at_ms'", (str(int(now_ms)),))

