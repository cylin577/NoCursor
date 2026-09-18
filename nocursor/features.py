"""Feature computation, normalization and clamping for cursor step vectors.

Each step of a resampled cursor episode is encoded as a fixed-size feature
vector that is screen-size invariant (relative geometry to target, deltas
normalized by a constant pixel scale).
"""

from __future__ import annotations

import math

import numpy as np

NORM_SCALE_PX = 100.0
DELTA_CLAMP = 1.5
TARGET_CLAMP = 1.5
DIST_CLAMP = 3.0
REACH_EPS_PX = 8.0
FEATURE_DIM = 7

FEATURE_NAMES = ("dx", "dy", "click", "dtx", "dty", "dist", "reached")

IDX_DX = 0
IDX_DY = 1
IDX_CLICK = 2
IDX_DTX = 3
IDX_DTY = 4
IDX_DIST = 5
IDX_REACHED = 6

# Indices of features that scale linearly with movement scale (used for
# augmentation jitter). dist is log-scaled so it is excluded.
SCALE_IDXS = (IDX_DX, IDX_DY, IDX_DTX, IDX_DTY)


def features_at(
    dx: float,
    dy: float,
    click: float,
    x: float,
    y: float,
    target: tuple[float, float],
) -> np.ndarray:
    """Feature vector of a single step.

    dx/dy: movement delta applied at this step (px; 0 at episode start).
    x, y: absolute position after the delta (px).
    target: (tx, ty) absolute target position (px).
    """
    tx, ty = target
    dist = math.hypot(tx - x, ty - y)
    return np.array(
        [
            float(np.clip(dx / NORM_SCALE_PX, -DELTA_CLAMP, DELTA_CLAMP)),
            float(np.clip(dy / NORM_SCALE_PX, -DELTA_CLAMP, DELTA_CLAMP)),
            float(click),
            float(np.clip((tx - x) / NORM_SCALE_PX, -TARGET_CLAMP, TARGET_CLAMP)),
            float(np.clip((ty - y) / NORM_SCALE_PX, -TARGET_CLAMP, TARGET_CLAMP)),
            float(min(math.log1p(dist / NORM_SCALE_PX), DIST_CLAMP)),
            1.0 if dist < REACH_EPS_PX else 0.0,
        ],
        dtype=np.float32,
    )


def episode_features(episode: dict) -> np.ndarray:
    """Feature matrix (T, FEATURE_DIM) of an episode dict.

    Episode dict must contain numpy arrays x, y, click, dx, dy and a
    target tuple. dx[0] must be 0 (start step has no movement yet).
    """
    x, y = episode["x"], episode["y"]
    dx, dy = episode["dx"], episode["dy"]
    click = episode["click"]
    target = tuple(episode["target"])
    out = np.empty((len(x), FEATURE_DIM), dtype=np.float32)
    for t in range(len(x)):
        out[t] = features_at(dx[t], dy[t], click[t], x[t], y[t], target)
    return out


def augment_features(f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random mirror flips and small scale jitter on a feature matrix."""
    g = f.copy()
    if rng.random() < 0.5:  # horizontal mirror
        g[:, IDX_DX] *= -1.0
        g[:, IDX_DTX] *= -1.0
    if rng.random() < 0.5:  # vertical mirror
        g[:, IDX_DY] *= -1.0
        g[:, IDX_DTY] *= -1.0
    scale = rng.uniform(0.9, 1.1)
    for i in SCALE_IDXS:
        g[:, i] *= scale
    return g
