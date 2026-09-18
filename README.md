
# NoCursor

Transformer-based human-like cursor movement synthesizer. Generates
realistic, human-like mouse trajectories (movement, timing, clicks)
conditioned on start/target — for GUI automation, demos and research.

## Architecture

Continuous-regression decoder-only transformer over per-step feature
vectors:

- **Input per step**: `[dx, dy, click, Δx_to_target, Δy_to_target,
  log(dist), target_reached]` — screen-size invariant (relative geometry)
- **Heads**: Gaussian `mu/logvar` per axis (stochastic sampling → human
  variability, not robotic average paths), plus click and target-reached
  binary heads
- **KV cache** for fast incremental generation
- Default: d_model=384, 8 layers, 8 heads, context 512 → ~14M params
  (~30MB FP16, far under the 2GB budget)

Training: Gaussian NLL + BCE, teacher forcing, AdamW + cosine schedule,
mirror-flip / scale-jitter augmentation, session-level train/val split.

## Quickstart

```bash
# setup
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev,play]'

# smoke test with synthetic data (no collector data needed)
python -m nocursor.train --synthetic --epochs 3 --d-model 128 --n-layers 4

# train on collected data
python -m nocursor.train --data data/collector.db --epochs 20

# synthesize a trajectory (CLI)
python -m nocursor.generate --checkpoint checkpoints/best.pt \
    --start 100,100 --target 800,500

# evaluate human-likeness vs held-out data
python -m nocursor.eval  # or: from nocursor.eval import evaluate

# fast runtime mode (INT8, same checkpoint)
python -m nocursor.quantize --checkpoint checkpoints/best.pt \
    --out checkpoints/fast.pt

# drive a live OS cursor (X11; see Wayland note below)
python -m nocursor.play --checkpoint checkpoints/fast.pt --mode fast \
    --random-targets
```

## Data collection

Web-based crowd collector: anonymous, consent-gated cursor capture.

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000/
```

- 24 Fitts-law target-clicking trials + 20s free movement
- Pointer Events with `getCoalescedEvents()` for high-fidelity capture
- Chunked upload to SQLite (`data/collector.db`, override with
  `NOCURSOR_DB`)

**Privacy**: explicit consent checkbox before any capture; anonymous UUID
session ids only; no PII, no raw IPs, no user agents, no keystrokes or
form contents collected. Data is gitignored.

## Pipeline

Raw events → resample to fixed 125 Hz grid (linear interpolation) →
segment into release-to-release episodes (movement + press, target =
press position) → normalized feature vectors → train.

## Known limitations

- **Wayland**: most compositors block synthetic cursor moves;
  `play.py` detects and warns. Use X11 or XWayland.
- Fitts eval fits MT vs log-distance only (target size not yet stored in
  episodes).
- `torch.ao.quantization.quantize_dynamic` is deprecated in newer PyTorch;
  works today, migrate to `torchao` when it drops.

## Layout

```
nocursor/          model.py, data.py, features.py, train.py, eval.py,
                   generate.py, play.py, quantize.py, synthdata.py
server/            FastAPI collector + static/ (consent page, task game)
tests/             pytest suite
scripts/           (empty)
```

## AI Declare
 Planed and executed by GLM-5.3-Flash 
I use AI bc im not familiar with Transformers lol 
