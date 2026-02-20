"""Long-context loss comparison across models (memory and baselines).

Runs all models on the same val windows at any context length.
W is always updating (normal inference) for memory models.

Usage:
    uv run eval_long.py --models model_mem1.pt model_wide.pt model_deep.pt
    uv run eval_long.py --models model_mem1.pt model_wide.pt --tokens 32768 --windows 2
"""

import argparse, math
import numpy as np
import torch
import torch.nn.functional as F

from data import load_dataset
from model import HebbianMamba


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = HebbianMamba(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    cfg = ckpt["config"]
    label = f"{path} ({n_params/1e6:.1f}M, d={cfg.d_model}, L={cfg.n_layers}, mem={cfg.use_memory})"
    return model, label


def detach(states):
    if isinstance(states, torch.Tensor):
        return states.detach()
    if isinstance(states, dict):
        return {k: detach(v) for k, v in states.items()}
    if isinstance(states, (list, tuple)):
        return type(states)(detach(v) for v in states)
    return states


@torch.no_grad()
def run_model(model, tokens, device):
    losses = []
    states = None
    for t in range(len(tokens) - 1):
        tok = torch.tensor([tokens[t]], device=device)
        tgt = torch.tensor([tokens[t + 1]], device=device)
        logits, states = model.step(tok, states=states)
        states = detach(states)
        losses.append(F.cross_entropy(logits, tgt).item())
    return np.array(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--tokens",  type=int, default=4096)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--segment", type=int, default=512)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | {args.tokens} tokens x {args.windows} windows\n")

    ds = load_dataset()
    rng = np.random.default_rng(42)
    max_start = len(ds["val"]) - args.tokens - 1
    starts = sorted(rng.choice(max_start, size=args.windows, replace=False))

    N, seg = args.tokens, args.segment
    n_segs = (N + seg - 1) // seg
    seg_labels = [f"{s}-{min(s+seg,N)}" for s in range(0, N, seg)]

    models = [load_model(path, device) for path in args.models]

    # all_losses[model_idx, window_idx, seg_idx]
    all_losses = np.zeros((len(models), args.windows, n_segs))

    for w_i, start in enumerate(starts):
        tokens = ds["val"][start : start + N + 1].tolist()
        print(f"Window {w_i+1}/{args.windows}: val[{start}:{start+N}]")
        for m_i, (model, label) in enumerate(models):
            losses = run_model(model, tokens, device)
            for si, s in enumerate(range(0, N, seg)):
                all_losses[m_i, w_i, si] = losses[s:min(s+seg,N)].mean()
            print(f"  {label}: {losses.mean():.4f}")
        print()

    # Summary table
    mean_loss = all_losses.mean(axis=1)   # (n_models, n_segs)
    overall   = all_losses.mean(axis=(1, 2))

    print(f"=== Overall (avg over {args.windows} windows) ===")
    col_w = 10
    header = f"{'Segment':>12}" + "".join(f"{f'model{i}':>{col_w}}" for i in range(len(models)))
    print(header)
    print("-" * (12 + col_w * len(models)))
    for si, lbl in enumerate(seg_labels):
        row = f"{lbl:>12}" + "".join(f"{mean_loss[mi, si]:>{col_w}.4f}" for mi in range(len(models)))
        print(row)
    print("-" * (12 + col_w * len(models)))
    row = f"{'OVERALL':>12}" + "".join(f"{overall[mi]:>{col_w}.4f}" for mi in range(len(models)))
    print(row)

    print("\nModels:")
    for i, (_, label) in enumerate(models):
        print(f"  model{i}: {label}")


if __name__ == "__main__":
    main()
