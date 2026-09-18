"""Export an INT8 dynamically-quantized model for the fast runtime mode.

Saves the whole quantized module object (quantized packed params do not
round-trip through state_dict without a matching quantize step).

Example:
    python -m nocursor.quantize --checkpoint checkpoints/best.pt \
        --out checkpoints/fast.pt
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

from nocursor.generate import load_checkpoint


def quantize(checkpoint_path: str):
    model = load_checkpoint(checkpoint_path)
    qmodel = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    qmodel.eval()
    return qmodel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    qmodel = quantize(args.checkpoint)
    torch.save(qmodel, args.out)
    print(f"quantized model saved to {args.out}")


if __name__ == "__main__":
    main()
