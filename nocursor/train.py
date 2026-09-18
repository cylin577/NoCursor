"""Train the cursor transformer on collector data.

Examples:
    python -m nocursor.train --data data/collector.db --epochs 20
    python -m nocursor.train --synthetic --epochs 3 --d-model 128 --n-layers 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nocursor.data import load_episodes_from_db, split_by_session
from nocursor.features import FEATURE_DIM, augment_features, episode_features
from nocursor.model import CursorTransformer, compute_loss


class EpisodeDataset(Dataset):
    def __init__(self, episodes: list[dict], augment: bool = False, seed: int = 0):
        self.features = [episode_features(ep) for ep in episodes]
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        f = self.features[i]
        if self.augment:
            f = augment_features(f, self.rng)
        # teacher forcing: input f[:-1], label f[1:] (predict next step)
        return torch.from_numpy(f[:-1]), torch.from_numpy(f[1:])


def collate(batch):
    xs, ys = zip(*batch)
    B = len(xs)
    Tmax = max(x.shape[0] for x in xs)
    D = xs[0].shape[1]
    x = torch.zeros(B, Tmax, D)
    y = torch.zeros(B, Tmax, D)
    mask = torch.zeros(B, Tmax, dtype=torch.bool)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        T = xi.shape[0]
        x[i, :T] = xi
        y[i, :T] = yi
        mask[i, :T] = True
    return x, y, mask


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        out = model(x)
        total += compute_loss(out, y, mask)["loss"].item() * x.size(0)
        count += x.size(0)
    return total / max(1, count)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=str, help="collector SQLite db path")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic dev data")
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    if args.synthetic:
        from nocursor.synthdata import build_synthetic_db

        db_path = os.path.join(args.out, "synthetic.db")
        os.makedirs(args.out, exist_ok=True)
        build_synthetic_db(db_path, n_sessions=8, seed=args.seed)
    elif args.data:
        db_path = args.data
    else:
        ap.error("provide --data or --synthetic")

    episodes = load_episodes_from_db(db_path)
    if not episodes:
        raise SystemExit(f"no episodes found in {db_path}")
    train_eps, val_eps = split_by_session(episodes, args.val_frac, args.seed)
    print(f"episodes: {len(episodes)} total, {len(train_eps)} train, {len(val_eps)} val")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CursorTransformer(
        feature_dim=FEATURE_DIM,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params on {device} "
          f"(~{n_params * 2 / 1e6:.0f}MB FP16)")

    train_ds = EpisodeDataset(train_eps, augment=True, seed=args.seed)
    val_ds = EpisodeDataset(val_eps, seed=args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup, total_steps)
    )

    os.makedirs(args.out, exist_ok=True)
    config = {
        "feature_dim": FEATURE_DIM,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "max_len": args.max_len,
    }
    best_val = float("inf")
    step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            out = model(x)
            losses = compute_loss(out, y, mask)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"loss {losses['loss'].item():.4f} "
                    f"nll {losses['nll'].item():.4f} "
                    f"click {losses['click'].item():.4f} "
                    f"reached {losses['reached'].item():.4f}"
                )
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch} done in {time.time() - t0:.1f}s, val loss {val_loss:.4f}")
        torch.save(
            {"model": model.state_dict(), "config": config},
            os.path.join(args.out, "last.pt"),
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model": model.state_dict(), "config": config, "val_loss": val_loss},
                os.path.join(args.out, "best.pt"),
            )
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"best val loss {best_val:.4f}, checkpoints in {args.out}/")


if __name__ == "__main__":
    main()
