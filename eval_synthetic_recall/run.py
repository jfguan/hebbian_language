"""Hebbian memory capacity benchmark.

Train on 32 KV pairs, then measure recall accuracy at increasing pair counts.
Plots the capacity curve: where does the d=128 memory matrix start to fail?
"""

import sys

sys.path.insert(0, "..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.hebbian_components import CausalConv, GatedMLP, HebbianBlock

N_KEYS = 256
N_VALS = 16
VOCAB = N_KEYS + N_VALS
TRAIN_PAIRS = 64


class Layer(nn.Module):
    def __init__(self, d_model, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.norm = nn.RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, expand=expand)
        self.conv = CausalConv(d_inner, d_conv=d_conv)
        self.memory = HebbianBlock(d_model, memory_alpha=0.03)

    def forward(self, x):
        normed = self.norm(x)
        out = self.mlp(normed, self.conv(self.mlp.project_up(normed)))
        out = self.memory(out)
        return x + out


class Model(nn.Module):
    def __init__(self, d_model, n_layers, d_conv=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model)
        self.layers = nn.ModuleList([Layer(d_model, d_conv=d_conv) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)
        self.emb.weight = self.head.weight

    def forward(self, ids):
        x = self.emb(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


def make_batch(B, pairs, device):
    T = 4 * pairs
    mid = T // 2
    x = torch.zeros(B, T, dtype=torch.long, device=device)

    keys = torch.stack([torch.randperm(N_KEYS, device=device)[:pairs] for _ in range(B)])
    vals = torch.randint(N_KEYS, VOCAB, (B, pairs), device=device)
    perms = torch.stack([torch.randperm(pairs, device=device) for _ in range(B)])

    idx = torch.arange(pairs, device=device)
    x[:, 2 * idx] = keys
    x[:, 2 * idx + 1] = vals
    x[:, mid + 2 * idx] = keys.gather(1, perms)
    x[:, mid + 2 * idx + 1] = vals.gather(1, perms)

    mask = torch.zeros(B, T - 1, device=device)
    mask[:, mid + 2 * idx] = 1.0
    return x, x[:, 1:], mask


def train(model, device, steps=1000, stop_acc=0.995):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(steps):
        x, tgt, mask = make_batch(64, TRAIN_PAIRS, device)
        logits = model(x)[:, :-1]

        loss = (
            F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
            .view_as(mask).mul(mask).sum() / mask.sum()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                acc = ((logits.argmax(-1) == tgt) * mask).sum() / mask.sum()
            print(f"  step {step:3d}  loss={loss.item():.4f}  recall={acc.item():.1%}")
            if acc.item() >= stop_acc:
                print(f"  converged at step {step}")
                return


def sweep(model, device):
    print("capacity sweep:")
    model.eval()
    results = {}
    with torch.no_grad():
        for p in [4, 8, 16, 32, 64, 96, 128, 160, 192, 256]:
            accs = []
            for _ in range(20):
                x, tgt, mask = make_batch(64, p, device)
                logits = model(x)[:, :-1]
                accs.append(((logits.argmax(-1) == tgt) * mask).sum() / mask.sum())
            results[p] = torch.stack(accs).mean().item()
            print(f"  {p:3d} pairs: {results[p]:.1%}")
    return results


def plot_capacity(results, path, d_model):
    pairs = list(results.keys())
    accs = [results[p] * 100 for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pairs, accs, "o-", linewidth=2, markersize=8)
    ax.axhline(y=100 / N_VALS, color="r", ls="--", alpha=0.5, label=f"chance ({100/N_VALS:.1f}%)")
    ax.set(
        xlabel="Number of KV pairs",
        ylabel="Recall accuracy (%)",
        title=f"Hebbian memory capacity (d_model={d_model})",
        ylim=(-5, 105),
    )
    ax.set_xticks(pairs)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    d_model = 128
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    model = Model(d_model=d_model, n_layers=5).to(device)
    print(f"device={device}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  train_pairs={TRAIN_PAIRS}")

    train(model, device)
    results = sweep(model, device)
    plot_capacity(results, "eval_synthetic_recall/capacity.png", d_model)
