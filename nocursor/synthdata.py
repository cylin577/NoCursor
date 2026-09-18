"""Synthetic human-like cursor dataset generator (development fallback).

Produces SQLite payloads in the exact collector format so the standard
data pipeline applies. Movements follow a minimum-jerk profile with Fitts's
law durations, occasional overshoot corrections and jittered sampling.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid

import numpy as np

from nocursor.data import SAMPLING_HZ

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


def min_jerk_pos(s: np.ndarray) -> np.ndarray:
    """Normalized minimum-jerk interpolation over s in [0, 1]."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def fitts_duration(dist_px: float, target_r_px: float, rng) -> float:
    """Fitts's law: MT = a + b * log2(2D/W)."""
    a, b = 0.18, 0.16
    return a + b * math.log2(2 * dist_px / (2 * target_r_px)) + rng.normal(0, 0.04)


def synthesize_session(
    rng: np.random.Generator,
    screen_w: int = 1280,
    screen_h: int = 800,
    n_trials: int = 24,
) -> list[dict]:
    """One synthetic session: Fitts target-clicking trials."""
    samples: list[dict] = []
    t = 0.0
    x, y = float(screen_w / 2), float(screen_h / 2)
    samples.append({"t": t, "x": x, "y": y, "type": "move"})

    for _ in range(n_trials):
        # target: random distance/angle from current cursor
        dist = rng.uniform(150.0, min(900.0, screen_w * 0.7))
        angle = rng.uniform(0, 2 * math.pi)
        target_r = rng.uniform(12.0, 40.0)
        tx = float(np.clip(x + dist * math.cos(angle), target_r, screen_w - target_r))
        ty = float(np.clip(y + dist * math.sin(angle), target_r, screen_h - target_r))
        actual_dist = math.hypot(tx - x, ty - y)

        # movement: minimum-jerk + noise + occasional overshoot correction
        mt = fitts_duration(actual_dist, target_r, rng)
        n_steps = max(3, int(mt * SAMPLING_HZ))
        s = np.linspace(0.0, 1.0, n_steps)
        base = min_jerk_pos(s)
        noise = np.cumsum(rng.normal(0, 0.004, n_steps))
        noise -= np.linspace(0, noise[-1], n_steps)  # anchor endpoints
        amp = 1.0
        if rng.random() < 0.3:  # overshoot then corrective sub-movement
            amp = 1.0 + rng.uniform(0.02, 0.08)
        path = x + (tx - x) * (base + noise) * amp
        path_y = y + (ty - y) * (base + noise) * amp
        if amp > 1.0:  # corrective tail back to target
            n_corr = max(3, int(0.08 * SAMPLING_HZ))
            s2 = np.linspace(0.0, 1.0, n_corr)
            path = np.concatenate([path, path[-1] + (tx - path[-1]) * min_jerk_pos(s2)])
            path_y = np.concatenate(
                [path_y, path_y[-1] + (ty - path_y[-1]) * min_jerk_pos(s2)]
            )

        for px, py in zip(path, path_y):
            t += (1000.0 / SAMPLING_HZ) * rng.uniform(0.7, 1.3)  # jittered events
            samples.append({"t": t, "x": float(px), "y": float(py), "type": "move"})

        # press + release
        x, y = float(tx), float(ty)
        press_ms = rng.uniform(60.0, 140.0)
        t += press_ms * 0.5
        samples.append({"t": t, "x": x, "y": y, "type": "down"})
        t += press_ms * 0.5
        samples.append({"t": t, "x": x, "y": y, "type": "up"})
        t += rng.uniform(150.0, 500.0)  # inter-trial rest

    samples.sort(key=lambda s: s["t"])
    return samples


def build_synthetic_db(path: str, n_sessions: int = 8, seed: int = 0) -> str:
    """Write synthetic sessions to a collector-format SQLite db."""
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        for _ in range(n_sessions):
            sid = str(uuid.uuid4())
            screen_w = int(rng.integers(1024, 1921))
            screen_h = int(rng.integers(640, 1081))
            samples = synthesize_session(rng, screen_w, screen_h)
            conn.execute(
                "INSERT INTO sessions (id, created_at, screen_w, screen_h, dpr, tasks, samples_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sid, time.time(), screen_w, screen_h, 1.0, json.dumps(["fitts"]), len(samples)),
            )
            conn.execute(
                "INSERT INTO chunks (session_id, seq, payload) VALUES (?, 0, ?)",
                (sid, json.dumps(samples)),
            )
        conn.commit()
    finally:
        conn.close()
    return path
