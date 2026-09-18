"""Decoder-only transformer over cursor-step feature vectors.

Input: sequence of per-step feature vectors (see features.py). Predicts the
next step's movement (dx, dy as Gaussian), click and target-reached (binary).
Supports incremental generation with a KV cache.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_D_MODEL = 384
DEFAULT_N_LAYERS = 8
DEFAULT_N_HEADS = 8
DEFAULT_MAX_LEN = 512

LOGVAR_MIN = -7.0
LOGVAR_MAX = 5.0


class LayerCache:
    """Per-layer KV cache for incremental decoding."""

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    @property
    def size(self) -> int:
        return 0 if self.k is None else int(self.k.size(-2))

    def append(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k if self.k is None else torch.cat([self.k, k], dim=-2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=-2)
        return self.k, self.v


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cache: LayerCache | None = None, pos_offset: int = 0):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            k, v = cache.append(k, v)

        tk = k.size(-2)
        q_pos = torch.arange(pos_offset, pos_offset + T, device=x.device)
        k_pos = torch.arange(tk, device=x.device)
        mask = k_pos[None, :] <= q_pos[:, None]  # (T, tk) True = attend
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~mask, float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.dropout(self.proj(out))
        x = x + self.mlp(self.ln2(x))
        return x


class CursorTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int = 7,
        d_model: int = DEFAULT_D_MODEL,
        n_layers: int = DEFAULT_N_LAYERS,
        n_heads: int = DEFAULT_N_HEADS,
        max_len: int = DEFAULT_MAX_LEN,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len

        self.tok = nn.Linear(feature_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head_mu_dx = nn.Linear(d_model, 1)
        self.head_lv_dx = nn.Linear(d_model, 1)
        self.head_mu_dy = nn.Linear(d_model, 1)
        self.head_lv_dy = nn.Linear(d_model, 1)
        self.head_click = nn.Linear(d_model, 1)
        self.head_reached = nn.Linear(d_model, 1)
        self.apply(self._init_weights)
        for head in (
            self.head_mu_dx,
            self.head_lv_dx,
            self.head_mu_dy,
            self.head_lv_dy,
            self.head_click,
            self.head_reached,
        ):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.01)

    def forward(self, x: torch.Tensor, caches: list[LayerCache] | None = None):
        """x: (B, T, feature_dim). caches: per-layer caches for generation.

        With caches, T must be 1 and pos_offset is derived from cache size.
        Returns dict of head outputs, each (B, T, 1).
        """
        B, T, _ = x.shape
        pos_offset = caches[0].size if caches is not None else 0
        assert pos_offset + T <= self.max_len, "sequence exceeds max_len"
        positions = torch.arange(pos_offset, pos_offset + T, device=x.device)
        h = self.tok(x) + self.pos(positions)[None, :, :]
        for i, block in enumerate(self.blocks):
            h = block(h, cache=caches[i] if caches is not None else None, pos_offset=pos_offset)
        h = self.ln_f(h)
        return {
            "mu_dx": self.head_mu_dx(h),
            "logvar_dx": self.head_lv_dx(h).clamp(LOGVAR_MIN, LOGVAR_MAX),
            "mu_dy": self.head_mu_dy(h),
            "logvar_dy": self.head_lv_dy(h).clamp(LOGVAR_MIN, LOGVAR_MAX),
            "click_logit": self.head_click(h),
            "reached_logit": self.head_reached(h),
        }


def compute_loss(
    outputs: dict,
    y: torch.Tensor,
    mask: torch.Tensor,
    w_nll: float = 1.0,
    w_click: float = 0.5,
    w_reached: float = 0.5,
) -> dict:
    """Multi-task loss. y: (B, T, feature_dim) next-step features,
    mask: (B, T) bool of valid positions.

    Movement: Gaussian NLL (mean + learned variance) -> stochastic
    sampling at inference instead of robotic average paths.
    """
    mask_f = mask.to(y.dtype)
    n = mask_f.sum().clamp(min=1.0)

    mu_dx = outputs["mu_dx"].squeeze(-1)
    lv_dx = outputs["logvar_dx"].squeeze(-1)
    mu_dy = outputs["mu_dy"].squeeze(-1)
    lv_dy = outputs["logvar_dy"].squeeze(-1)
    nll_dx = 0.5 * (lv_dx + (y[..., 0] - mu_dx) ** 2 * torch.exp(-lv_dx))
    nll_dy = 0.5 * (lv_dy + (y[..., 1] - mu_dy) ** 2 * torch.exp(-lv_dy))
    nll = (nll_dx + nll_dy) * mask_f
    loss_nll = nll.sum() / n

    bce_click = F.binary_cross_entropy_with_logits(
        outputs["click_logit"].squeeze(-1), y[..., 2], reduction="none"
    )
    bce_reached = F.binary_cross_entropy_with_logits(
        outputs["reached_logit"].squeeze(-1), y[..., 6], reduction="none"
    )
    loss_click = (bce_click * mask_f).sum() / n
    loss_reached = (bce_reached * mask_f).sum() / n

    total = w_nll * loss_nll + w_click * loss_click + w_reached * loss_reached
    return {
        "loss": total,
        "nll": loss_nll.detach(),
        "click": loss_click.detach(),
        "reached": loss_reached.detach(),
    }
