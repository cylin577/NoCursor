"""Human-likeness evaluation metrics.

Compares generated episodes against held-out human episodes:
- Fitts's law fit (movement time vs log-distance, R^2)
- Velocity profile shape (Pearson correlation of mean normalized profiles)
- Jerk statistics (mean normalized jerk ratio)
- Overshoot rate
- DTW distance between speed profiles
"""

from __future__ import annotations

import math

import numpy as np


def movement_time(ep: dict) -> float:
    """Steps until the press starts, in seconds at the episode sample rate."""
    press = np.flatnonzero(ep["click"] > 0.5)
    if len(press) == 0:
        return float(len(ep["x"])) / float(ep.get("hz", 125.0))
    return float(press[0]) / float(ep.get("hz", 125.0))


def travel_distance(ep: dict) -> float:
    """Start-to-target straight-line distance."""
    x0, y0 = ep["x"][0], ep["y"][0]
    tx, ty = ep["target"]
    return float(math.hypot(tx - x0, ty - y0))


def fitts_fit(episodes: list[dict]) -> dict:
    """Linear fit MT ~ a + b*log2(D+1). Returns slope, R^2.

    Uses distance only (target size is not stored in episode dicts);
    R^2 close to human values indicates Fitts-like scaling.
    """
    d = np.array([travel_distance(ep) for ep in episodes])
    mt = np.array([movement_time(ep) for ep in episodes])
    keep = d > 0
    d, mt = d[keep], mt[keep]
    if len(d) < 3:
        return {"slope": float("nan"), "r2": float("nan"), "n": int(len(d))}
    x = np.log2(d + 1.0)
    A = np.vstack([np.ones_like(x), x]).T
    coef, res, _, _ = np.linalg.lstsq(A, mt, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((mt - pred) ** 2))
    ss_tot = float(np.sum((mt - mt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": float(coef[1]), "r2": r2, "n": int(len(d))}


def speed_curve(episodes: list[dict], n_points: int = 50) -> np.ndarray:
    """Mean normalized speed profile: speed vs normalized time."""
    curves = []
    for ep in episodes:
        dx, dy = ep["dx"], ep["dy"]
        speed = np.hypot(dx, dy)
        t = np.linspace(0.0, 1.0, n_points)
        grid = np.linspace(0.0, len(speed), n_points)
        interp = np.interp(grid, np.arange(len(speed)), speed)
        peak = interp.max()
        if peak > 0:
            curves.append(interp / peak)
    if not curves:
        return np.zeros(n_points)
    return np.mean(curves, axis=0)


def profile_correlation(human: np.ndarray, model: np.ndarray) -> float:
    """Pearson r between two mean speed profiles."""
    if human.std() == 0 or model.std() == 0:
        return float("nan")
    return float(np.corrcoef(human, model)[0, 1])


def mean_jerk(episodes: list[dict]) -> float:
    """Mean normalized jerk: |third derivative| integrated over the path,
    normalized by duration (higher = jerkier movement)."""
    jerks = []
    for ep in episodes:
        dx, dy = ep["dx"], ep["dy"]
        if len(dx) < 4:
            continue
        vx, vy = np.diff(dx), np.diff(dy)
        ax, ay = np.diff(vx), np.diff(vy)
        jx, jy = np.diff(ax), np.diff(ay)
        jerk = np.mean(np.hypot(jx, jy))
        duration = len(dx) / float(ep.get("hz", 125.0))
        if duration > 0:
            jerks.append(jerk / duration)
    return float(np.mean(jerks)) if jerks else float("nan")


def overshoot_rate(episodes: list[dict]) -> float:
    """Fraction of episodes whose path crosses past the target along the
    start->target axis before arriving (overshoot + correction)."""
    count, total = 0, 0
    for ep in episodes:
        x0, y0 = ep["x"][0], ep["y"][0]
        tx, ty = ep["target"]
        d = np.array([tx - x0, ty - y0])
        norm = np.hypot(*d)
        if norm < 1e-6:
            continue
        u = d / norm
        proj = (ep["x"] - x0) * u[0] + (ep["y"] - y0) * u[1]
        total += 1
        if proj.max() > norm + 2.0:  # >2px past target
            count += 1
    return count / total if total else float("nan")


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Dynamic time warping distance between two 1D speed profiles."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def mean_dtw(human_eps: list[dict], model_eps: list[dict], n_points: int = 50) -> float:
    """Mean DTW between per-episode speed profiles (sampled pairs)."""
    rng = np.random.default_rng(0)
    pairs = min(len(human_eps), len(model_eps), 100)
    if pairs == 0:
        return float("nan")
    hi = rng.choice(len(human_eps), pairs)
    mi = rng.choice(len(model_eps), pairs)
    dists = []
    for h, m in zip(hi, mi):
        hs = _resampled_speed(human_eps[int(h)], n_points)
        ms = _resampled_speed(model_eps[int(m)], n_points)
        dists.append(dtw_distance(hs, ms))
    return float(np.mean(dists))


def _resampled_speed(ep: dict, n_points: int) -> np.ndarray:
    speed = np.hypot(ep["dx"], ep["dy"])
    grid = np.linspace(0.0, len(speed), n_points)
    return np.interp(grid, np.arange(len(speed)), speed)


def evaluate(human_eps: list[dict], model_eps: list[dict] | None = None) -> dict:
    """Full human-likeness report."""
    out = {
        "human": {
            "n": len(human_eps),
            "fitts": fitts_fit(human_eps),
            "jerk": mean_jerk(human_eps),
            "overshoot": overshoot_rate(human_eps),
        }
    }
    if model_eps:
        out["model"] = {
            "n": len(model_eps),
            "fitts": fitts_fit(model_eps),
            "jerk": mean_jerk(model_eps),
            "overshoot": overshoot_rate(model_eps),
        }
        out["profile_correlation"] = profile_correlation(
            speed_curve(human_eps), speed_curve(model_eps)
        )
        out["dtw"] = mean_dtw(human_eps, model_eps)
    return out
