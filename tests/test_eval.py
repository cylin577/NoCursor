import numpy as np

from nocursor.data import resample, segment_episodes
from nocursor.eval import (
    evaluate,
    fitts_fit,
    mean_jerk,
    overshoot_rate,
    profile_correlation,
    speed_curve,
)
from nocursor.synthdata import synthesize_session


def make_episodes(n_sessions=3, seed=0):
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n_sessions):
        eps.extend(segment_episodes(resample(synthesize_session(rng))))
    return eps


def test_fitts_fit_strong_on_synthetic():
    eps = make_episodes()
    fit = fitts_fit(eps)
    assert fit["n"] > 0
    assert 0.0 < fit["r2"] <= 1.0  # synthetic data is Fitts-generated


def test_metrics_finite():
    eps = make_episodes()
    assert np.isfinite(mean_jerk(eps))
    assert 0.0 <= overshoot_rate(eps) <= 1.0
    curve = speed_curve(eps)
    assert curve.shape == (50,)
    assert np.isfinite(curve).all()


def test_evaluate_report():
    human = make_episodes(n_sessions=3, seed=0)
    model = make_episodes(n_sessions=3, seed=1)
    report = evaluate(human, model)
    assert "human" in report and "model" in report
    assert -1.0 <= report["profile_correlation"] <= 1.0
    assert np.isfinite(report["dtw"])
