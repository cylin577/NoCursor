import numpy as np
import torch

from nocursor.model import (
    CursorTransformer,
    LayerCache,
    compute_loss,
)


def make_model():
    torch.manual_seed(0)
    return CursorTransformer(
        feature_dim=7, d_model=64, n_layers=2, n_heads=4, max_len=64
    )


def seq(B=2, T=16):
    torch.manual_seed(1)
    return torch.randn(B, T, 7)


def test_forward_shapes():
    model = make_model()
    out = model(seq())
    for v in out.values():
        assert v.shape == (2, 16, 1)


def test_causal_masking_no_future_leak():
    model = make_model()
    x = seq()
    out_full = model(x)
    # perturb future steps; past outputs must not change
    x2 = x.clone()
    x2[:, 10:] += 10.0
    out_pert = model(x2)
    assert torch.allclose(out_full["mu_dx"][:, :10], out_pert["mu_dx"][:, :10])
    assert not torch.allclose(out_full["mu_dx"][:, 10:], out_pert["mu_dx"][:, 10:])


def test_kv_cache_matches_full_forward():
    model = make_model()
    model.eval()
    x = seq(B=1, T=20)
    with torch.no_grad():
        out_full = model(x)
        caches = [LayerCache() for _ in model.blocks]
        outs = []
        for t in range(20):
            o = model(x[:, t : t + 1], caches=caches)
            outs.append({k: v[:, -1] for k, v in o.items()})
    for k in out_full:
        full = out_full[k].squeeze(-1)  # (1, 20)
        inc = torch.cat([o[k] for o in outs], dim=1).squeeze(-1)  # (1, 20)
        assert torch.allclose(full, inc, atol=1e-5), f"cache mismatch on {k}"


def test_cache_size_grows():
    model = make_model()
    caches = [LayerCache() for _ in model.blocks]
    x = seq(B=1, T=5)
    model(x, caches=caches)
    assert all(c.size == 5 for c in caches)


def test_compute_loss_finite_and_positive():
    model = make_model()
    x = seq()
    y = torch.clamp(x, -1.5, 1.5) * 0.1
    y[..., 2] = (torch.rand_like(y[..., 2]) > 0.5).float()
    y[..., 6] = 0.0
    mask = torch.ones(2, 16, dtype=torch.bool)
    losses = compute_loss(model(x), y, mask)
    assert torch.isfinite(losses["loss"])
    assert losses["loss"].item() > 0


def test_loss_backward():
    model = make_model()
    x = seq()
    y = torch.zeros(2, 16, 7)
    mask = torch.ones(2, 16, dtype=torch.bool)
    losses = compute_loss(model(x), y, mask)
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
