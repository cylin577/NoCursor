import numpy as np
import torch

from nocursor.generate import CursorSynthesizer, load_checkpoint
from nocursor.model import CursorTransformer


def make_model():
    torch.manual_seed(0)
    model = CursorTransformer(feature_dim=7, d_model=64, n_layers=2, n_heads=4, max_len=64)
    model.eval()
    return model


def test_synthesizer_steps_and_done():
    model = make_model()
    synth = CursorSynthesizer(model, seed=0, max_steps=50)
    synth.reset((100.0, 100.0), (110.0, 100.0))
    steps = 0
    while True:
        dx, dy, click, done = synth.step()
        steps += 1
        assert steps <= 50
        if done:
            break
    assert 0 < steps <= 50


def test_synthesizer_deterministic_with_seed():
    model = make_model()
    runs = []
    for _ in range(2):
        synth = CursorSynthesizer(model, seed=42, max_steps=20)
        synth.reset((0.0, 0.0), (500.0, 300.0))
        trace = []
        while True:
            dx, dy, click, done = synth.step()
            trace.append((dx, dy, click, done))
            if done:
                break
        runs.append(trace)
    assert runs[0] == runs[1]


def test_synthesizer_respects_max_len():
    model = make_model()  # max_len=64
    synth = CursorSynthesizer(model, seed=1, temperature=0.0)
    synth.reset((0.0, 0.0), (50000.0, 50000.0))  # unreachable target
    for _ in range(64):
        dx, dy, click, done = synth.step()
        if done:
            break
    assert synth.step_count <= 64


def test_checkpoint_roundtrip(tmp_path):
    model = make_model()
    path = str(tmp_path / "m.pt")
    torch.save(
        {"model": model.state_dict(),
         "config": {"feature_dim": 7, "d_model": 64, "n_layers": 2,
                    "n_heads": 4, "max_len": 64}},
        path,
    )
    loaded = load_checkpoint(path)
    assert isinstance(loaded, CursorTransformer)
    a = CursorSynthesizer(loaded, seed=7, temperature=0.0)
    a.reset((0.0, 0.0), (300.0, 200.0))
    b = CursorSynthesizer(model, seed=7, temperature=0.0)
    b.reset((0.0, 0.0), (300.0, 200.0))
    for _ in range(10):
        da, _, _, done_a = a.step()
        db, _, _, done_b = b.step()
        assert da == db
        if done_a or done_b:
            break
