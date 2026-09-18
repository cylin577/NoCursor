"""Drive a live OS cursor with the synthesizer.

Dual runtime modes from one checkpoint:
- quality: FP32/FP16 weights, full context, full sampling temperature
- fast:    INT8 dynamically quantized model

Example:
    python -m nocursor.play --checkpoint checkpoints/best.pt --mode quality
    python -m nocursor.play --checkpoint checkpoints/fast.pt --mode fast \
        --random-targets --max-episodes 5
"""

from __future__ import annotations

import argparse
import math
import os
import time

from nocursor.generate import CursorSynthesizer, load_checkpoint

PLAY_HZ = 125.0


def check_display() -> str | None:
    """Warn about Wayland: synthetic cursor moves are X11-only in practice."""
    session = os.environ.get("XDG_SESSION_TYPE", "")
    if session.lower() == "wayland":
        return (
            "Wayland session detected: most Wayland compositors block synthetic "
            "cursor moves. Playback may fail or do nothing. Use an X11 session "
            "or XWayland."
        )
    return None


def pick_random_target(rng, x, y, screen_w, screen_h, margin=60):
    while True:
        dist = rng.uniform(150.0, min(900.0, max(screen_w, screen_h) * 0.7))
        angle = rng.uniform(0, 2 * math.pi)
        tx = x + dist * math.cos(angle)
        ty = y + dist * math.sin(angle)
        if margin < tx < screen_w - margin and margin < ty < screen_h - margin:
            return float(tx), float(ty)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--mode", choices=["quality", "fast"], default="quality")
    ap.add_argument("--rate", type=float, default=PLAY_HZ, help="steps per second")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--random-targets", action="store_true",
                    help="wander to random screen targets (with clicks)")
    ap.add_argument("--targets", type=str, default="",
                    help="comma-separated x,y pairs, e.g. 800,500 200,300")
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = unlimited")
    ap.add_argument("--screen-w", type=int, default=1280)
    ap.add_argument("--screen-h", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    warning = check_display()
    if warning:
        print(f"WARNING: {warning}")

    try:
        from pynput.mouse import Button, Controller
    except Exception as e:  # pragma: no cover - depends on display
        raise SystemExit(
            f"pynput unavailable ({e}). Install with: pip install 'nocursor[play]' "
            "and run under a display server (X11)."
        )

    if args.mode == "fast" and args.checkpoint.endswith("best.pt"):
        print("note: fast mode is intended with an INT8 export "
              "(python -m nocursor.quantize)")

    model = load_checkpoint(args.checkpoint)
    synth = CursorSynthesizer(model, temperature=args.temperature, seed=args.seed)

    targets = []
    if args.targets:
        for pair in args.targets.split():
            tx, ty = (float(v) for v in pair.split(","))
            targets.append((tx, ty))

    mouse = Controller()
    pos = mouse.position
    print(f"mode={args.mode} rate={args.rate:.0f}Hz start=({pos[0]:.0f},{pos[1]:.0f})")
    print("Ctrl+C to stop")

    rng = np.random.default_rng(args.seed)
    period = 1.0 / args.rate
    episode_idx = 0
    target = None
    try:
        while True:
            if target is None:
                if targets:
                    target = targets.pop(0)
                elif args.random_targets:
                    target = pick_random_target(rng, pos[0], pos[1],
                                                args.screen_w, args.screen_h)
                else:
                    raise SystemExit("provide --targets or --random-targets")
                synth.reset(pos, target)
                episode_idx += 1
                if args.max_episodes and episode_idx > args.max_episodes:
                    print("done")
                    break
                print(f"episode {episode_idx}: target {target}")

            next_tick = time.perf_counter() + period
            dx, dy, click, done = synth.step()
            if dx or dy:
                mouse.move(dx, dy)
                pos = (pos[0] + dx, pos[1] + dy)
            if click:
                mouse.press(Button.left)
                mouse.release(Button.left)
            if done:
                target = None
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
