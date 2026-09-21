"""Transactional experiment checkpoints; images are never stored in SQLite."""

from __future__ import annotations

import json
import sqlite3
import time
import zlib
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any


def pack(value: Any) -> bytes:
    return zlib.compress(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(), 6)


def unpack(value: bytes) -> Any:
    return json.loads(zlib.decompress(value))


class RobustDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs(
                    id TEXT PRIMARY KEY, config BLOB NOT NULL, manifest BLOB NOT NULL,
                    status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                    heartbeat REAL NOT NULL DEFAULT 0, pid INTEGER, cancel INTEGER DEFAULT 0,
                    elapsed REAL DEFAULT 0, message TEXT DEFAULT '', environment BLOB,
                    report BLOB);
                CREATE TABLE IF NOT EXISTS tasks(
                    run_id TEXT, image TEXT, provider TEXT, view TEXT, status TEXT,
                    elapsed REAL, result BLOB, error TEXT,
                    PRIMARY KEY(run_id,image,provider,view));
                CREATE TABLE IF NOT EXISTS pages(
                    run_id TEXT, image TEXT, status TEXT, payload BLOB,
                    PRIMARY KEY(run_id,image));
                CREATE TABLE IF NOT EXISTS events(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, stamp REAL, message TEXT);
                CREATE INDEX IF NOT EXISTS events_run ON events(run_id,seq);
                CREATE TABLE IF NOT EXISTS page_metrics(
                    run_id TEXT,image TEXT,payload BLOB,PRIMARY KEY(run_id,image));
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db

    def create(self, identifier: str, config: dict, manifest: dict, environment: dict) -> None:
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM runs WHERE status IN ('queued','running','cancelling')").fetchone():
                raise ValueError("A robustness experiment is already active")
            db.execute("INSERT INTO runs(id,config,manifest,status,created,updated,environment) VALUES(?,?,?,?,?,?,?)",
                       (identifier, pack(config), pack(manifest), "queued", now, now, pack(environment)))

    def get(self, identifier: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise ValueError("Unknown robustness experiment")
            result = dict(row)
            for key in ("config", "manifest", "environment", "report"):
                result[key] = unpack(result[key]) if result[key] else None
            result["tasks_done"] = db.execute("SELECT COUNT(*) FROM tasks WHERE run_id=? AND status='done'", (identifier,)).fetchone()[0]
            result["task_errors"] = db.execute("SELECT COUNT(*) FROM tasks WHERE run_id=? AND status='error'", (identifier,)).fetchone()[0]
            result["pages_done"] = db.execute("SELECT COUNT(*) FROM pages WHERE run_id=?", (identifier,)).fetchone()[0]
            result["completed_images"] = [r[0] for r in db.execute("SELECT image FROM pages WHERE run_id=? AND status='done' ORDER BY image", (identifier,))]
            result["events"] = [dict(x) for x in reversed(db.execute("SELECT * FROM events WHERE run_id=? ORDER BY seq DESC LIMIT 80", (identifier,)).fetchall())]
        result["tasks_total"] = len(result["manifest"]["images"]) * len(result["config"]["providers"]) * len(result["config"]["views"])
        finished = result["tasks_done"] + result["task_errors"]
        result["eta_seconds"] = result["elapsed"] / finished * (result["tasks_total"]-finished) if finished else None
        return result

    def list(self) -> list[dict]:
        with self.connect() as db:
            identifiers = [x[0] for x in db.execute("SELECT id FROM runs ORDER BY created DESC")]
        return [self.get(identifier) for identifier in identifiers]

    def update(self, identifier: str, **values: Any) -> None:
        allowed = {"status", "pid", "heartbeat", "cancel", "elapsed", "message", "report"}
        if set(values) - allowed:
            raise ValueError("Unknown run fields")
        if "report" in values:
            values["report"] = pack(values["report"])
        values["updated"] = time.time()
        with self.connect() as db:
            db.execute("UPDATE runs SET " + ",".join(f"{k}=?" for k in values) + " WHERE id=?",
                       (*values.values(), identifier))

    def event(self, identifier: str, message: str) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO events(run_id,stamp,message) VALUES(?,?,?)", (identifier, time.time(), message))
            db.execute("UPDATE runs SET message=?,updated=? WHERE id=?", (message, time.time(), identifier))

    def cancelled(self, identifier: str) -> bool:
        with self.connect() as db:
            return bool(db.execute("SELECT cancel FROM runs WHERE id=?", (identifier,)).fetchone()[0])

    def claim_resume(self, identifier: str) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM runs WHERE status IN ('queued','running','cancelling')").fetchone():
                raise ValueError("An experiment is already active")
            db.execute("UPDATE runs SET status='queued',cancel=0,updated=? WHERE id=?", (time.time(), identifier))

    def task(self, identifier: str, image: str, provider: str, view: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE run_id=? AND image=? AND provider=? AND view=?",
                             (identifier, image, provider, view)).fetchone()
        if row is None:
            return None
        return {**dict(row), "result": unpack(row["result"]) if row["result"] else None}

    def save_task(self, identifier: str, image: str, provider: str, view: str,
                  elapsed: float, result: dict | None, error: str = "") -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?,?)",
                       (identifier, image, provider, view, "error" if error else "done", elapsed,
                        pack(result) if result is not None else None, error))

    def save_page(self, identifier: str, image: str, payload: dict, status: str = "done") -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO pages VALUES(?,?,?,?)", (identifier, image, status, pack(payload)))
            compact = {k: payload[k] for k in ("image", "has_gt", "complete", "metrics", "view_metrics", "stability", "size")}
            db.execute("INSERT OR REPLACE INTO page_metrics VALUES(?,?,?)", (identifier, image, pack(compact)))

    def metric_pages(self, identifier: str) -> list[dict]:
        with self.connect() as db:
            return [unpack(row[0]) for row in db.execute("SELECT payload FROM page_metrics WHERE run_id=? ORDER BY image", (identifier,))]

    def page(self, identifier: str, image: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM pages WHERE run_id=? AND image=?", (identifier, image)).fetchone()
        return unpack(row[0]) if row else None

    def pages(self, identifier: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM pages WHERE run_id=? ORDER BY image", (identifier,)).fetchall()
        return [unpack(row[0]) for row in rows]
