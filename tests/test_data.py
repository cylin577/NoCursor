import numpy as np
import pytest

from nocursor.data import (
    build_episode,
    parse_samples,
    resample,
    segment_episodes,
    split_by_session,
)


def make_samples():
    """2s of movement to (400, 200), press, 0.5s rest, short move, press."""
    samples = []
    t = 0.0
    for i in range(126):  # ~1s at 125Hz
        t += 8.0
        samples.append({"t": t, "x": 400.0 * i / 125, "y": 200.0 * i / 125, "type": "move"})
    t += 8.0
    samples.append({"t": t, "x": 400.0, "y": 200.0, "type": "move"})
    t += 50.0
    samples.append({"t": t, "x": 400.0, "y": 200.0, "type": "down"})
    t += 80.0
    samples.append({"t": t, "x": 400.0, "y": 200.0, "type": "up"})
    t += 500.0  # rest gap
    for i in range(40):
        t += 8.0
        samples.append({"t": t, "x": 400.0 + 3 * i, "y": 200.0, "type": "move"})
    t += 50.0
    samples.append({"t": t, "x": 517.0, "y": 200.0, "type": "down"})
    t += 80.0
    samples.append({"t": t, "x": 517.0, "y": 200.0, "type": "up"})
    samples.sort(key=lambda s: s["t"])
    return samples


def test_parse_samples_drops_bad():
    raw = make_samples() + [{"t": 1.0, "x": 1.0, "type": "bad"}, {"t": "x", "y": 1}, {"t": 2.0, "x": 5.0, "y": 5.0, "type": "move"}]
    parsed = parse_samples(raw)
    assert all(s["type"] in ("move", "down", "up") for s in parsed)
    ts = [s["t"] for s in parsed]
    assert ts == sorted(ts)


def test_resample_grid_and_click():
    res = resample(make_samples())
    assert res is not None
    n = len(res["x"])
    assert abs(n - 261) < 20  # ~2.1s session at 125Hz
    assert res["click"].sum() > 0  # press captured
    # click only in press regions (not during first second)
    assert res["click"][:100].sum() == 0


def test_segment_episodes():
    res = resample(make_samples())
    eps = segment_episodes(res)
    assert len(eps) == 2
    # first episode: to (400, 200); target = press position
    assert abs(eps[0]["target"][0] - 400.0) < 3.0
    assert eps[0]["click"][0] == 0.0  # starts before press
    assert eps[0]["click"][-1] == 1.0  # ends during press
    assert eps[0]["dx"][0] == 0.0  # start step has no delta


def test_segment_drops_tiny():
    res = resample(make_samples())
    eps = build_episode(res, 0, 2)  # 3 steps < MIN_EPISODE_STEPS
    assert eps is None


def test_split_by_session():
    base = segment_episodes(resample(make_samples()))
    eps = [dict(ep) for _ in range(10) for ep in base]
    train, val = split_by_session(eps, val_frac=0.2, seed=0)
    assert len(train) + len(val) == len(eps)
    assert len(val) >= 1
    ids = [id(ep) for ep in train] + [id(ep) for ep in val]
    assert len(ids) == len(set(ids))  # no leakage


def test_resample_empty():
    assert resample([]) is None
    assert resample([{"t": 0.0, "x": 0.0, "y": 0.0, "type": "move"}]) is None
