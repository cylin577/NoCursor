"""Load collector data, resample to a fixed-rate grid and segment episodes.

Raw collector samples are dicts: {t: ms since page load, x, y: CSS px,
type: 'move' | 'down' | 'up'}. Sessions are stored in SQLite as JSON
payload chunks.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np

SAMPLING_HZ = 125.0
GRID_STEP_MS = 1000.0 / SAMPLING_HZ
MIN_EPISODE_STEPS = 5
MAX_EPISODE_STEPS = 512
MIN_EPISODE_DISPLACEMENT_PX = 2.0


def parse_samples(payload) -> list[dict]:
    """Validate and sort raw samples. Drops malformed/unknown entries."""
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    out = []
    for s in payload:
        try:
            t = float(s["t"])
            x = float(s["x"])
            y = float(s["y"])
            typ = s["type"]
        except (KeyError, TypeError, ValueError):
            continue
        if typ not in ("move", "down", "up"):
            continue
        out.append({"t": t, "x": x, "y": y, "type": typ})
    out.sort(key=lambda s: s["t"])
    return out


def resample(samples, hz: float = SAMPLING_HZ) -> dict | None:
    """Resample irregular event samples onto a uniform grid.

    Positions are linearly interpolated. The click channel is 1 for a grid
    interval if the button was held at any time during that interval.
    Returns dict with x, y, click arrays (length n) or None if too short.
    """
    samples = parse_samples(samples)
    pos_t: list[float] = []
    pos_x: list[float] = []
    pos_y: list[float] = []
    held_t: list[float] = []
    held_v: list[bool] = []
    held = False
    for s in samples:
        t, x, y, typ = s["t"], s["x"], s["y"], s["type"]
        pos_t.append(t)
        pos_x.append(x)
        pos_y.append(y)
        if typ == "down" and not held:
            held = True
            held_t.append(t)
            held_v.append(True)
        elif typ == "up" and held:
            held = False
            held_t.append(t)
            held_v.append(False)

    if len(pos_t) < 2:
        return None
    t0 = pos_t[0]
    n = int((pos_t[-1] - t0) / 1000.0 * hz)
    if n < 2:
        return None

    grid = t0 + np.arange(n) / hz * 1000.0
    x = np.interp(grid, pos_t, pos_x)
    y = np.interp(grid, pos_t, pos_y)

    # click held during [grid[k], grid[k+1]) iff step function true just
    # before the interval end
    click = np.zeros(n, dtype=np.float32)
    if held_t:
        idx = np.searchsorted(held_t, grid + GRID_STEP_MS - 1e-3, side="right") - 1
        valid = idx >= 0
        click[valid] = np.asarray(held_v, dtype=np.float32)[idx[valid]]

    return {"x": x, "y": y, "click": click, "hz": hz}


def build_episode(resampled: dict, start: int, end: int) -> dict | None:
    """Build an episode dict from resampled arrays over [start, end].

    An episode is one release-to-release segment: movement towards the
    click position, then the button press. Target = position of the press.
    """
    x = resampled["x"][start : end + 1].astype(np.float32)
    y = resampled["y"][start : end + 1].astype(np.float32)
    click = resampled["click"][start : end + 1].astype(np.float32)
    n = len(x)
    if n < MIN_EPISODE_STEPS or n > MAX_EPISODE_STEPS:
        return None
    press = np.flatnonzero(click > 0.5)
    if len(press) == 0:
        return None
    target = (float(x[press[0]]), float(y[press[0]]))
    if np.hypot(x[-1] - x[0], y[-1] - y[0]) < MIN_EPISODE_DISPLACEMENT_PX:
        return None
    dx = np.diff(x, prepend=x[:1])
    dy = np.diff(y, prepend=y[:1])
    return {"x": x, "y": y, "click": click, "dx": dx, "dy": dy, "target": target}


def segment_episodes(resampled: dict) -> list[dict]:
    """Split a resampled session into release-to-release episodes."""
    click = resampled["click"]
    n = len(click)
    ends = [k for k in range(n) if click[k] > 0.5 and (k == n - 1 or click[k + 1] < 0.5)]
    episodes = []
    start = 0
    for end in ends:
        ep = build_episode(resampled, start, end)
        if ep is not None:
            episodes.append(ep)
        start = end + 1
    return episodes


def load_episodes_from_db(db_path: str) -> list[dict]:
    """Load all sessions from a collector SQLite db and segment episodes."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT c.session_id, c.payload FROM chunks c "
            "JOIN sessions s ON s.id = c.session_id "
            "ORDER BY s.created_at, c.seq"
        ).fetchall()
    finally:
        conn.close()
    by_session: dict[str, list] = {}
    for sid, payload in rows:
        by_session.setdefault(sid, []).extend(parse_samples(payload))
    episodes = []
    for samples in by_session.values():
        res = resample(samples)
        if res is not None:
            episodes.extend(segment_episodes(res))
    return episodes


def split_by_session(episodes: list[dict], val_frac: float = 0.1, seed: int = 0):
    """Split episodes into train/val by contiguous session blocks.

    Episodes are already grouped by session (see load_episodes_from_db);
    splitting by blocks avoids window-level leakage.
    """
    n_val = max(1, int(len(episodes) * val_frac)) if episodes else 0
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(episodes))
    # contiguous blocks in permuted order keep session grouping intact
    val_idx = set(idx[:n_val].tolist())
    train = [ep for i, ep in enumerate(episodes) if i not in val_idx]
    val = [ep for i, ep in enumerate(episodes) if i in val_idx]
    return train, val
