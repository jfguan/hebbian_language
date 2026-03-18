"""Hebbian memory ablation: W updating vs W frozen.

Runs inference on val data, comparing normal inference (W accumulates)
against a frozen baseline (W reset each token). The delta shows how much
the memory matrix contributes at different context depths.
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import load_dataset
from models import build_model

# -- config --
TOKENS = 8192
WINDOWS = 4
SEGMENT = 1024
DATASET = "the_stack"
CHECKPOINT = "checkpoints/hebbian_18M_the_stack.pt"


def main():
    device = "mps"
    model = setup(device)
    dataset = load_dataset(DATASET)

    segment_losses = evaluate_windows(model, dataset, device)
    print_results(segment_losses)

    plot_results(segment_losses, "eval_results/memory.png")


def setup(device):
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(
        f"device={device}  model={checkpoint['model_config'].name}  tokens={TOKENS}  windows={WINDOWS}"
    )
    return model


def evaluate_windows(model, dataset, device):
    """Compare W-updating vs W-frozen across random val windows.

    Each window gets two passes: one normal (memory accumulates),
    one frozen (memory reset each step). Per-segment averages reveal
    how much the memory contributes at each context depth.
    """
    n_segments = TOKENS // SEGMENT
    rng = np.random.default_rng(42)

    # pick non-overlapping windows from val set
    max_start = len(dataset.val) - TOKENS - 1
    starts = sorted(rng.choice(max_start, size=WINDOWS, replace=False))

    updating = np.zeros((WINDOWS, n_segments))
    frozen = np.zeros((WINDOWS, n_segments))

    for i, start in enumerate(starts):
        tokens = dataset.val[start : start + TOKENS + 1].tolist()
        print(f"\nwindow {i + 1}/{WINDOWS}: val[{start}:{start + TOKENS}]")

        # two passes: normal vs frozen memory
        loss_normal = run_pass(model, tokens, device, freeze_memory=False)
        loss_frozen = run_pass(model, tokens, device, freeze_memory=True)

        # average loss per segment
        for s in range(n_segments):
            lo, hi = s * SEGMENT, (s + 1) * SEGMENT
            updating[i, s] = loss_normal[lo:hi].mean()
            frozen[i, s] = loss_frozen[lo:hi].mean()

        delta = loss_frozen.mean() - loss_normal.mean()
        print(f"  updating={loss_normal.mean():.4f}  frozen={loss_frozen.mean():.4f}  delta={delta:+.4f}")

    return {"updating": updating, "frozen": frozen}


@torch.no_grad()
def run_pass(model, tokens, device, freeze_memory=False):
    """Run sequential inference, optionally freezing W after each step."""
    N = len(tokens) - 1
    losses = np.zeros(N)
    states = None

    for t in range(N):
        token = torch.tensor([tokens[t]], device=device)
        target = torch.tensor([tokens[t + 1]], device=device)

        if freeze_memory and states is not None:
            saved_W = [s["memory"]["W"].clone() for s in states]

        logits, states = model.step(token, states=states)
        states = [detach(s) for s in states]
        losses[t] = F.cross_entropy(logits, target).item()

        if freeze_memory and t > 0:
            for i in range(len(states)):
                states[i]["memory"]["W"] = saved_W[i]

    return losses


def detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, dict):
        return {k: detach(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(detach(v) for v in x)
    return x


def print_results(segment_losses):
    updating = segment_losses["updating"]
    frozen = segment_losses["frozen"]
    deltas = frozen - updating
    n_segments = updating.shape[1]

    mean_updating = updating.mean(axis=0)
    mean_frozen = frozen.mean(axis=0)
    mean_delta = deltas.mean(axis=0)
    stderr_delta = deltas.std(axis=0) / math.sqrt(WINDOWS)

    overall_delta = deltas.mean()
    overall_stderr = deltas.std() / math.sqrt(WINDOWS * n_segments)

    print(f"\n=== Aggregated over {WINDOWS} windows ===")
    print(
        f"Overall: updating={updating.mean():.4f}  frozen={frozen.mean():.4f}  delta={overall_delta:+.4f} +/- {overall_stderr:.4f}"
    )
    print(
        f"\n{'Segment':>12}  {'Updating':>10}  {'Frozen':>10}  {'Delta':>10}  {'Stderr':>8}"
    )
    print("-" * 56)
    for seg_idx, seg_start in enumerate(range(0, TOKENS, SEGMENT)):
        seg_end = min(seg_start + SEGMENT, TOKENS)
        print(
            f"{seg_start:>5}-{seg_end:<5}  {mean_updating[seg_idx]:>10.4f}  {mean_frozen[seg_idx]:>10.4f}  {mean_delta[seg_idx]:>+10.4f}  {stderr_delta[seg_idx]:>8.4f}"
        )


def plot_results(segment_losses, path):
    updating = segment_losses["updating"].mean(axis=0)
    frozen = segment_losses["frozen"].mean(axis=0)
    segments = [f"{s}-{min(s + SEGMENT, TOKENS)}" for s in range(0, TOKENS, SEGMENT)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(segments))
    ax.bar(x - 0.15, updating, 0.3, label="W updating", alpha=0.8)
    ax.bar(x + 0.15, frozen, 0.3, label="W frozen", alpha=0.8)
    ax.set(
        xlabel="Segment", ylabel="CE Loss", title="Memory Contribution by Context Depth"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(segments, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_memory_params(model, path):
    layers = range(len(model.layers))
    decays = [torch.sigmoid(l.memory.decay).mean().item() for l in model.layers]
    alphas = [l.memory.log_alpha.exp().mean().item() for l in model.layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(layers, decays, alpha=0.8)
    ax1.set(
        xlabel="Layer",
        ylabel="Decay (gamma)",
        title="Learned Decay per Layer",
        ylim=(0, 1),
    )
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(layers, alphas, alpha=0.8, color="tab:orange")
    ax2.set(xlabel="Layer", ylabel="Alpha", title="Learned Alpha per Layer")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
