"""Inspect Hebbian memory parameters and track W behavior during inference."""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    return model


def detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, dict):
        return {k: detach(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(detach(v) for v in x)
    return x


def print_param_table(model):
    print("\n=== Hebbian Memory Parameters ===")
    print(f"{'Layer':>5}  {'σ(decay)':>9}  {'‖proj_w‖':>9}  {'‖proj_r‖':>9}")
    print("-" * 39)
    for i, layer in enumerate(model.layers):
        if not layer.use_memory:
            print(f"{i:>5}  (no memory)")
            continue
        gamma = torch.sigmoid(layer.decay).item()
        pw = layer.proj_write.weight.data.norm().item()
        pr = layer.proj_read.weight.data.norm().item()
        print(f"{i:>5}  {gamma:>9.6f}  {pw:>9.4f}  {pr:>9.4f}")
    print(f"\nFixed α = 0.03")


@torch.no_grad()
def run_inference(model, tokens, device):
    """Two passes: W updating (normal) vs W frozen (zero throughout).
    For no-memory models, only runs one pass (both are identical)."""
    N = len(tokens) - 1
    L = len(model.layers)
    has_memory = any(layer.use_memory for layer in model.layers)

    w_norms = np.zeros((L, N))
    loss_upd = np.zeros(N)

    # Pass 1: normal — W accumulates across tokens
    states = None
    for t in range(N):
        tok = torch.tensor([tokens[t]], device=device)
        tgt = torch.tensor([tokens[t + 1]], device=device)
        logits, states = model.step(tok, states=states)
        states = [detach(s) for s in states]
        loss_upd[t] = F.cross_entropy(logits, tgt).item()
        if has_memory:
            for i in range(L):
                mem = states[i]["memory"]
                w_norms[i, t] = mem.norm().item() if mem is not None else 0.0

    if not has_memory:
        return w_norms, loss_upd, loss_upd  # no W means both passes identical

    # Pass 2: frozen — W restored to pre-step value each token
    loss_frz = np.zeros(N)
    states = None
    for t in range(N):
        tok = torch.tensor([tokens[t]], device=device)
        tgt = torch.tensor([tokens[t + 1]], device=device)
        if states is not None:
            saved_W = [s["memory"].clone() for s in states]
        logits, states = model.step(tok, states=states)
        states = [detach(s) for s in states]
        loss_frz[t] = F.cross_entropy(logits, tgt).item()
        if t > 0:
            for i in range(L):
                states[i]["memory"] = saved_W[i]

    return w_norms, loss_upd, loss_frz


def plot_results(w_norms, loss_upd, loss_frz, path):
    L, N = w_norms.shape
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    boundaries = [b for b in range(512, N, 512)]

    # W norms per layer
    colors = plt.cm.viridis(np.linspace(0, 1, L))
    for i in range(L):
        ax1.plot(w_norms[i], color=colors[i], alpha=0.7, label=f"layer {i}")
    for b in boundaries:
        ax1.axvline(b, color="red", ls="--", alpha=0.4,
                     label="512-tok boundary" if b == boundaries[0] else None)
    ax1.set(ylabel="‖W‖_F", title="W Frobenius Norm (W updating)")
    ax1.legend(fontsize=7, ncol=4)
    ax1.grid(True, alpha=0.3)

    # Smoothed loss comparison
    win = 50
    if N >= win:
        kernel = np.ones(win) / win
        x = np.arange(win - 1, N)
        ax2.plot(x, np.convolve(loss_upd, kernel, mode="valid"), label="W updating", alpha=0.8)
        ax2.plot(x, np.convolve(loss_frz, kernel, mode="valid"), label="W frozen", alpha=0.8)
    else:
        ax2.plot(loss_upd, label="W updating", alpha=0.8)
        ax2.plot(loss_frz, label="W frozen", alpha=0.8)
    for b in boundaries:
        ax2.axvline(b, color="red", ls="--", alpha=0.4)
    ax2.set(xlabel="Token position", ylabel=f"CE Loss (smoothed, w={win})",
            title="Per-Token Loss: W Updating vs W Frozen")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--model", type=str, default="checkpoints/model_mem1.pt")
    p.add_argument("--segment", type=int, default=512)
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(args.model, device)
    ds = load_dataset()

    print_param_table(model)

    N = args.tokens
    n_windows = args.windows
    seg = args.segment
    n_segs = (N + seg - 1) // seg

    # Pick random non-overlapping windows from val
    rng = np.random.default_rng(42)
    max_start = len(ds["val"]) - N - 1
    starts = sorted(rng.choice(max_start, size=n_windows, replace=False))

    # Collect per-segment deltas across windows
    all_seg_upd = np.zeros((n_windows, n_segs))
    all_seg_frz = np.zeros((n_windows, n_segs))
    last_w_norms, last_loss_upd, last_loss_frz = None, None, None

    for w_i, start in enumerate(starts):
        tokens = ds["val"][start : start + N + 1].tolist()
        print(f"\nWindow {w_i+1}/{n_windows}: val[{start}:{start+N}]")

        w_norms, loss_upd, loss_frz = run_inference(model, tokens, device)
        last_w_norms, last_loss_upd, last_loss_frz = w_norms, loss_upd, loss_frz

        for si, s in enumerate(range(0, N, seg)):
            e = min(s + seg, N)
            all_seg_upd[w_i, si] = loss_upd[s:e].mean()
            all_seg_frz[w_i, si] = loss_frz[s:e].mean()

        avg_upd, avg_frz = loss_upd.mean(), loss_frz.mean()
        print(f"  Avg loss: upd={avg_upd:.4f} frz={avg_frz:.4f} delta={avg_frz-avg_upd:+.4f}")

    # Aggregate across windows
    deltas = all_seg_frz - all_seg_upd
    mean_upd = all_seg_upd.mean(axis=0)
    mean_frz = all_seg_frz.mean(axis=0)
    mean_delta = deltas.mean(axis=0)
    stderr_delta = deltas.std(axis=0) / math.sqrt(n_windows)

    print(f"\n=== Aggregated over {n_windows} windows ===")
    print(f"Overall: upd={all_seg_upd.mean():.4f} frz={all_seg_frz.mean():.4f} delta={deltas.mean():+.4f} ± {deltas.std()/math.sqrt(n_windows*n_segs):.4f}")
    print(f"\n{'Segment':>12}  {'Loss(upd)':>10}  {'Loss(frz)':>10}  {'Delta':>10}  {'±stderr':>8}")
    print("-" * 56)
    for si, s in enumerate(range(0, N, seg)):
        e = min(s + seg, N)
        print(f"{s:>5}-{e:<5}  {mean_upd[si]:>10.4f}  {mean_frz[si]:>10.4f}  {mean_delta[si]:>+10.4f}  {stderr_delta[si]:>8.4f}")

    plot_results(last_w_norms, last_loss_upd, last_loss_frz, "eval_memory/eval_memory.png")


if __name__ == "__main__":
    main()
