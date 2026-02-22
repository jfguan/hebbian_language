"""Generate code training curves for paper — two subplots: 18M and 100M."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(tag, tps):
    entries = [json.loads(l) for l in open(f"checkpoints/history_{tag}.jsonl") if "val_loss" in l]
    tokens = [e["tokens"] / 1e6 if e.get("tokens", 0) > 0 else e["step"] * tps / 1e6
              for e in entries]
    vals = [e["val_loss"] for e in entries]
    return tokens, vals


# --- 18M figure ---
fig1, ax1 = plt.subplots(figsize=(7, 4))
tps_18m = 4096
for tag, label, color, ls in [
    ("code_memory", "Memory (d=512, 8L)",        "#1f77b4", "-"),
    ("code_deep",   "Deep baseline (d=512, 10L)", "#2ca02c", "--"),
]:
    t, v = load(tag, tps_18m)
    ax1.plot(t, v, ls, color=color, label=label, linewidth=1.8, marker="o", markersize=3)
ax1.set_xlabel("Training tokens (M)")
ax1.set_ylabel("Val loss (nats)")
ax1.set_title("Python code val loss — 18M parameters")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig("figures/code_training_18M.pdf", dpi=150)
fig1.savefig("figures/code_training_18M.png", dpi=150)
print("Saved figures/code_training_18M.pdf and .png")

# --- 100M figure ---
fig2, ax2 = plt.subplots(figsize=(7, 4))
tps_100m = 2048
for tag, label, color, ls in [
    ("code100M_memory", "Memory (d=1024, 12L)",        "#1f77b4", "-"),
    ("code100M_deep",   "Deep baseline (d=1024, 16L)", "#2ca02c", "--"),
]:
    t, v = load(tag, tps_100m)
    ax2.plot(t, v, ls, color=color, label=label, linewidth=1.8, marker="o", markersize=3)
ax2.set_xlabel("Training tokens (M)")
ax2.set_ylabel("Val loss (nats)")
ax2.set_title("Python code val loss — 100M parameters")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("figures/code_training_100M.pdf", dpi=150)
fig2.savefig("figures/code_training_100M.png", dpi=150)
print("Saved figures/code_training_100M.pdf and .png")
