"""NoCursor web data collector server.

Anonymous, consent-gated cursor movement collection. Serves the collector
page and stores sessions in SQLite. No PII: anonymous UUID session ids, no
raw IPs, no user agents, no form-field contents.

Run:
    uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from nocursor.data import parse_samples

DB_PATH = os.environ.get("NOCURSOR_DB", "data/collector.db")
STATIC_DIR = Path(__file__).parent / "static"

MAX_CHUNK_SAMPLES = 5000
MAX_SESSION_CHUNKS = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    screen_w INTEGER NOT NULL,
    screen_h INTEGER NOT NULL,
    dpr REAL NOT NULL,
    tasks TEXT NOT NULL,
    samples_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);
"""


def get_db() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


app = FastAPI(title="NoCursor collector", docs_url=None, redoc_url=None)


class SessionMeta(BaseModel):
    screen_w: int = Field(ge=1, le=20000)
    screen_h: int = Field(ge=1, le=20000)
    dpr: float = Field(ge=0.1, le=10.0)
    tasks: list[str] = Field(default_factory=list)
    consent: bool

    @field_validator("consent")
    @classmethod
    def consent_must_be_true(cls, v):
        if v is not True:
            raise ValueError("explicit consent required before any data collection")
        return v


class SampleChunk(BaseModel):
    samples: list[dict]

    @field_validator("samples")
    @classmethod
    def limit_size(cls, v):
        if len(v) > MAX_CHUNK_SAMPLES:
            raise ValueError(f"chunk too large (max {MAX_CHUNK_SAMPLES} samples)")
        return v


def _no_cache(path: Path):
    return FileResponse(path, headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    return _no_cache(STATIC_DIR / "index.html")


@app.get("/app.js")
def app_js():
    return _no_cache(STATIC_DIR / "app.js")


@app.post("/api/sessions")
def create_session(meta: SessionMeta):
    sid = str(uuid.uuid4())
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO sessions (id, created_at, screen_w, screen_h, dpr, tasks) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, time.time(), meta.screen_w, meta.screen_h, meta.dpr,
             json.dumps(meta.tasks)),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": sid}


@app.post("/api/sessions/{sid}/chunks")
def append_chunk(sid: str, chunk: SampleChunk):
    conn = get_db()
    try:
        row = conn.execute("SELECT id FROM sessions WHERE id = ?", (sid,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="unknown session")
        count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        if count >= MAX_SESSION_CHUNKS:
            raise HTTPException(status_code=413, detail="session full")
        cleaned = parse_samples(chunk.samples)
        if not cleaned:
            return {"stored": 0}
        conn.execute(
            "INSERT INTO chunks (session_id, seq, payload) VALUES (?, ?, ?)",
            (sid, count, json.dumps(cleaned)),
        )
        conn.execute(
            "UPDATE sessions SET samples_count = samples_count + ? WHERE id = ?",
            (len(cleaned), sid),
        )
        conn.commit()
    finally:
        conn.close()
    return {"stored": len(cleaned)}


@app.get("/api/stats")
def stats():
    conn = get_db()
    try:
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        samples = conn.execute(
            "SELECT COALESCE(SUM(samples_count), 0) FROM sessions"
        ).fetchone()[0]
    finally:
        conn.close()
    return {"sessions": sessions, "samples": samples}
