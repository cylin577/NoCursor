"""Cursor movement synthesis API.

Example:
    python -m nocursor.generate --checkpoint checkpoints/best.pt \
        --start 100,100 --target 800,500 --steps-print 10
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from nocursor.features import DELTA_CLAMP, NORM_SCALE_PX, REACH_EPS_PX, features_at
from nocursor.model import CursorTransformer, LayerCache


class CursorSynthesizer:
    """Autoregressive step-by-step synthesis with KV cache."""

    def __init__(
        self,
        model: CursorTransformer,
        device: str = "cpu",
        temperature: float = 1.0,
        reach_eps_px: float = REACH_EPS_PX,
        max_steps: int | None = None,
        seed: int | None = None,
    ):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.reach_eps_px = reach_eps_px
        self.max_steps = max_steps or model.max_len
        self.rng = np.random.default_rng(seed)

    def reset(self, start_xy: tuple[float, float], target_xy: tuple[float, float]):
        self.x, self.y = float(start_xy[0]), float(start_xy[1])
        self.target = (float(target_xy[0]), float(target_xy[1]))
        self.caches = [LayerCache() for _ in self.model.blocks]
        self.step_count = 0
        self._done = False
        self._last_dx = 0.0
        self._last_dy = 0.0
        self._last_click = 0.0

    @torch.no_grad()
    def _predict(self, feat: np.ndarray) -> dict:
        x = torch.from_numpy(feat)[None, None, :].to(self.device)
        out = self.model(x, caches=self.caches)
        return {k: v[0, -1, 0].item() for k, v in out.items()}

    @torch.no_grad()
    def step(self) -> tuple[float, float, float, bool]:
        """Advance one step. Returns (dx, dy, click, done) in px / binary."""
        if self._done:
            return 0.0, 0.0, 0.0, True

        # current step input: no movement yet at this step (delta applied
        # after prediction, mirroring training where dx[t] is fed before
        # predicting dx[t+1])
        dx_now = 0.0 if self.step_count == 0 else self._last_dx
        feat = features_at(dx_now, self._last_dy if self.step_count else 0.0,
                           self._last_click if self.step_count else 0.0,
                           self.x, self.y, self.target)
        pred = self._predict(feat)

        mu, lv = pred["mu_dx"], pred["logvar_dx"]
        sigma = math.exp(0.5 * lv)
        dx = mu + sigma * self.rng.normal(0.0, 1.0) * self.temperature
        mu_y, lv_y = pred["mu_dy"], pred["logvar_dy"]
        sigma_y = math.exp(0.5 * lv_y)
        dy = mu_y + sigma_y * self.rng.normal(0.0, 1.0) * self.temperature
        dx = float(np.clip(dx, -DELTA_CLAMP, DELTA_CLAMP)) * NORM_SCALE_PX
        dy = float(np.clip(dy, -DELTA_CLAMP, DELTA_CLAMP)) * NORM_SCALE_PX

        click = 1.0 if self.rng.random() < _sigmoid(pred["click_logit"]) else 0.0
        self.x += dx
        self.y += dy
        self._last_dx, self._last_dy, self._last_click = dx, dy, click
        self.step_count += 1

        dist = np.hypot(self.target[0] - self.x, self.target[1] - self.y)
        reached_head = _sigmoid(pred["reached_logit"]) > 0.5
        if dist < self.reach_eps_px or reached_head or self.step_count >= self.max_steps:
            self._done = True
        return dx, dy, click, self._done


def _sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-v))


def load_checkpoint(path: str, device: str = "cpu"):
    """Load a checkpoint saved by train.py (or quantize.py whole-object save)."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, dict) and "config" in blob:
        model = CursorTransformer(**blob["config"])
        model.load_state_dict(blob["model"])
    else:
        model = blob  # quantized whole-object save
    model.eval().to(device)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--start", type=str, default="100,100", help="x,y in px")
    ap.add_argument("--target", type=str, default="800,500", help="x,y in px")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steps-print", type=int, default=10, help="print first N steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    start = tuple(float(v) for v in args.start.split(","))
    target = tuple(float(v) for v in args.target.split(","))
    model = load_checkpoint(args.checkpoint)
    synth = CursorSynthesizer(model, temperature=args.temperature, seed=args.seed)
    synth.reset(start, target)

    total_dx = total_dy = 0.0
    while True:
        dx, dy, click, done = synth.step()
        total_dx += dx
        total_dy += dy
        if synth.step_count <= args.steps_print or done:
            print(
                f"step {synth.step_count:4d}: dx={dx:7.2f} dy={dy:7.2f} "
                f"click={click:.0f} pos=({synth.x:7.1f},{synth.y:7.1f})"
            )
        if done:
            break
    print(f"done: {synth.step_count} steps, final pos ({synth.x:.1f},{synth.y:.1f}), "
          f"target {target}")


if __name__ == "__main__":
    main()
