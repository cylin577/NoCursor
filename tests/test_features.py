import numpy as np
import pytest

from nocursor.features import (
    DELTA_CLAMP,
    FEATURE_DIM,
    REACH_EPS_PX,
    augment_features,
    episode_features,
    features_at,
)


def make_episode(n=20, target=(300.0, 200.0)):
    x = np.linspace(0.0, target[0], n, dtype=np.float32)
    y = np.linspace(0.0, target[1], n, dtype=np.float32)
    dx = np.diff(x, prepend=x[:1])
    dy = np.diff(y, prepend=y[:1])
    click = np.zeros(n, dtype=np.float32)
    click[-3:] = 1.0
    return {"x": x, "y": y, "click": click, "dx": dx, "dy": dy, "target": target}


def test_features_at_shape_and_start():
    f = features_at(0.0, 0.0, 0.0, 0.0, 0.0, (300.0, 200.0))
    assert f.shape == (FEATURE_DIM,)
    assert f[0] == 0.0  # dx zero at start
    assert f[6] == 0.0  # not reached


def test_features_at_reached():
    f = features_at(0.0, 0.0, 1.0, 300.0, 200.0, (300.0, 200.0))
    assert f[6] == 1.0  # dist < eps -> reached
    assert f[5] < np.log1p(REACH_EPS_PX / 100.0) + 1e-6


def test_features_delta_clamped():
    f = features_at(5000.0, -5000.0, 0.0, 5000.0, -5000.0, (0.0, 0.0))
    assert f[0] == DELTA_CLAMP
    assert f[1] == -DELTA_CLAMP


def test_episode_features_matrix():
    ep = make_episode()
    f = episode_features(ep)
    assert f.shape == (20, FEATURE_DIM)
    # dx matches actual per-step movement
    np.testing.assert_allclose(f[1:, 0], np.diff(ep["x"]) / 100.0, atol=1e-6)
    # dtx shrinks towards 0 as cursor approaches target
    assert f[-1, 3] < f[0, 3]


def test_augment_mirror_flips():
    rng = np.random.default_rng(0)
    ep = make_episode()
    f = episode_features(ep)
    flipped = None
    for _ in range(20):
        g = augment_features(f, rng)
        if g[5, 0] != f[5, 0] and g[5, 1] == f[5, 1]:
            flipped = g  # horizontal flip
            break
    if flipped is not None:
        assert np.allclose(flipped[:, 0], -f[:, 0])
        assert np.allclose(flipped[:, 3], -f[:, 3])


def test_augment_preserves_click():
    rng = np.random.default_rng(1)
    f = episode_features(make_episode())
    for _ in range(10):
        g = augment_features(f, rng)
        np.testing.assert_array_equal(g[:, 2], f[:, 2])
