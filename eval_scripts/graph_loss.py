"""Graph training loss curves for 100M models with fitted lines."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SMOOTH_WINDOW = 500
OUT_PATH = "eval_results/loss_curves.png"

MODELS = [
    ("histories/gdn_100M_the_stack.jsonl", "GDN 100M"),
    ("histories/delta_hebbian_100M_the_stack.jsonl", "Delta Hebbian 100M"),
    ("histories/swa_delta_100M_the_stack.jsonl", "SWA Delta 100M"),
    ("histories/dual_delta_100M_the_stack.jsonl", "Dual Delta 100M"),
]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def main():
    fig, ax = plt.subplots(figsize=(12, 6))

    for (path, label), color in zip(MODELS, COLORS):
        tokens, loss = load_history(path)
        if len(tokens) == 0:
            continue

        smoothed, smoothed_tokens = smooth(loss, tokens)

        ax.plot(smoothed_tokens, smoothed, "-", color=color, linewidth=1.5, alpha=0.25)

        # -- fit line (power law on smoothed data) --
        mask = smoothed_tokens > 0
        log_t, log_l = np.log(smoothed_tokens[mask]), np.log(smoothed[mask])
        coeffs = np.polyfit(log_t, log_l, 1)
        fit_tokens = np.linspace(
            smoothed_tokens[mask].min(), smoothed_tokens[mask].max(), 200
        )
        fit_loss = np.exp(np.polyval(coeffs, np.log(fit_tokens)))
        # -- label: smoothed loss at 100M tokens --
        idx_100M = np.searchsorted(smoothed_tokens, 100.0)
        idx_100M = min(idx_100M, len(smoothed) - 1)
        loss_100M = smoothed[idx_100M]

        ax.plot(
            fit_tokens,
            fit_loss,
            "-",
            color=color,
            linewidth=2.5,
            label=f"{label} ({loss_100M:.3f} @100M)",
        )

    ax.set(xlabel="Tokens (M)", ylabel="Train Loss", title="100M Training Curves")
    ax.set_ylim(0.8, 4.0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved {OUT_PATH}")


def load_history(path):
    entries = [json.loads(line) for line in open(path) if "train_loss" in line]
    tokens = np.array([e["tokens"] / 1e6 for e in entries])
    loss = np.array([e["train_loss"] for e in entries])
    return tokens, loss


def smooth(values, x):
    w = min(SMOOTH_WINDOW, len(values))
    if w <= 1:
        return values, x
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode="valid"), x[w - 1 :]


if __name__ == "__main__":
    main()
