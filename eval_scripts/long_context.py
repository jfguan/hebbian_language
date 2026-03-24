"""Long-context eval: per-segment loss across models.

Usage:
    uv run eval_scripts/long_context.py checkpoints/swa_delta_100M_the_stack.pt checkpoints/delta_hebbian_100M_the_stack.pt
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import load_dataset
from models import build_model

TOKENS = 8192
WINDOWS = 4
SEGMENT = 1024
DATASET = "the_stack"
OUT_PATH = "eval_results/long_context.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", help="checkpoint paths")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset = load_dataset(DATASET)
    models = [load_model(path, device) for path in args.checkpoints]
    windows = pick_windows(dataset)

    print(f"device={device}  tokens={TOKENS}  windows={WINDOWS}  models={len(models)}")

    losses = evaluate(models, dataset, windows, device)
    print_table(losses, models)
    plot(losses, models)


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    name = os.path.basename(path).replace(".pt", "")
    return model, name


def pick_windows(dataset):
    rng = np.random.default_rng(42)
    max_start = len(dataset.val) - TOKENS - 1
    return sorted(rng.choice(max_start, size=WINDOWS, replace=False))


def evaluate(models, dataset, windows, device):
    """Returns (n_models, n_segments) array of mean losses."""
    n_segments = TOKENS // SEGMENT
    all_losses = np.zeros((len(models), WINDOWS, n_segments))

    for w, start in enumerate(windows):
        tokens = dataset.val[start : start + TOKENS + 1].tolist()
        print(f"\nwindow {w + 1}/{WINDOWS}: val[{start}:{start + TOKENS}]")

        for m, (model, name) in enumerate(models):
            per_token = run_sequential(model, tokens, device)
            for s in range(n_segments):
                lo, hi = s * SEGMENT, (s + 1) * SEGMENT
                all_losses[m, w, s] = per_token[lo:hi].mean()
            print(f"  {name}: {per_token.mean():.4f}")

    return all_losses.mean(axis=1)  # average over windows


@torch.no_grad()
def run_sequential(model, tokens, device):
    losses = np.zeros(len(tokens) - 1)
    states = None
    for t in range(len(tokens) - 1):
        token = torch.tensor([tokens[t]], device=device)
        target = torch.tensor([tokens[t + 1]], device=device)
        logits, states = model.step(token, states=states)
        states = detach(states)
        losses[t] = F.cross_entropy(logits, target).item()
    return losses


def detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, dict):
        return {k: detach(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(detach(v) for v in x)
    return x


def print_table(losses, models):
    names = [name for _, name in models]
    n_segments = losses.shape[1]
    col_w = max(len(n) for n in names) + 2

    print(f"\n{'Segment':>10}" + "".join(f"{n:>{col_w}}" for n in names))
    print("-" * (10 + col_w * len(names)))
    for s in range(n_segments):
        label = f"{s * SEGMENT // 1024}K-{(s + 1) * SEGMENT // 1024}K"
        vals = "".join(f"{losses[m, s]:>{col_w}.4f}" for m in range(len(names)))
        print(f"{label:>10}{vals}")
    print("-" * (10 + col_w * len(names)))
    overall = "".join(f"{losses[m].mean():>{col_w}.4f}" for m in range(len(names)))
    print(f"{'OVERALL':>10}{overall}")


def plot(losses, models):
    names = [name for _, name in models]
    n_segments = losses.shape[1]
    segments = [f"{s}-{s + SEGMENT // 1024}K" for s in range(0, TOKENS // 1024, SEGMENT // 1024)]
    x = np.arange(n_segments)

    fig, ax = plt.subplots(figsize=(12, 5))
    for m, name in enumerate(names):
        ax.plot(x, losses[m], "o-", markersize=5, linewidth=2, label=f"{name} ({losses[m].mean():.3f})")

    ax.set(xlabel="Segment", ylabel="Loss (nats)", title=f"Per-Segment Loss ({TOKENS // 1024}K context)")
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=45, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT_PATH}")


if __name__ == "__main__":
    main()
