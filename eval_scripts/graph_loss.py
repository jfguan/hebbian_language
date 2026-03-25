"""Graph training loss curves for 100M models."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import build_model

OUT_PATH = "eval_results/loss_curves.png"
MODELS = [
    ("histories/gdn_100M_the_stack.jsonl",            "checkpoints/gdn_100M_the_stack.pt",            "GDN"),
    ("histories/delta_hebbian_100M_the_stack.jsonl",   "checkpoints/delta_hebbian_100M_the_stack.pt",   "Delta Hebbian"),
    ("histories/swa_delta_100M_the_stack.jsonl",       "checkpoints/swa_delta_100M_the_stack.pt",       "SWA Delta"),
    ("histories/dual_delta_100M_the_stack.jsonl",      "checkpoints/dual_delta_100M_the_stack.pt",      "Dual Delta"),
]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def main():
    fig, ax = plt.subplots(figsize=(12, 6))

    for (hist, ckpt, name), color in zip(MODELS, COLORS):
        tokens, loss = load_history(hist)
        params_M = model_params(ckpt)
        smooth_t, smooth_l = smooth(tokens, loss)
        ax.plot(smooth_t, smooth_l, "-", color=color, linewidth=1, alpha=0.2)

        # power law fit in log-log space, skip early training
        mask = smooth_t > 25
        log_t, log_l = np.log(smooth_t[mask]), np.log(smooth_l[mask])
        coeffs = np.polyfit(log_t, log_l, 1)
        fit_t = np.linspace(smooth_t.min(), smooth_t.max(), 300)
        fit_l = np.exp(np.polyval(coeffs, np.log(fit_t)))
        ax.plot(fit_t, fit_l, "-", color=color, linewidth=2, label=f"{name} ({params_M:.0f}M)")

    ax.set(xlabel="Tokens (M)", ylabel="Train Loss (nats)", title="Training Loss")
    ax.set_ylim(1.0, 3.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved {OUT_PATH}")


SMOOTH_WINDOW = 500


def load_history(path):
    entries = [json.loads(l) for l in open(path) if "train_loss" in l]
    return (np.array([e["tokens"] / 1e6 for e in entries]),
            np.array([e["train_loss"] for e in entries]))


def smooth(x, y):
    w = min(SMOOTH_WINDOW, len(y))
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


def model_params(ckpt_path):
    cfg = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_config"]
    cfg.vocab_size = 1024
    return sum(p.numel() for p in build_model(cfg).parameters()) / 1e6


if __name__ == "__main__":
    main()
