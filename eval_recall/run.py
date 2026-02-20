"""Hebbian memory capacity benchmark.

Train on 32 KV pairs, then measure recall accuracy at [4, 8, 16, 32, 64, 128, 256].
Plots the capacity curve: where does the d=128 memory matrix start to fail?
"""

import sys
sys.path.insert(0, "..")

from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mambapy.mamba import MambaBlock, MambaConfig as MambaCfg, RMSNorm
from model import Config

N_KEYS = 256  # enough unique keys for 256 pairs
N_VALS = 16   # small prediction space so the model can actually learn
VOCAB = N_KEYS + N_VALS
TRAIN_PAIRS = 32


# -- Model --

class Layer(nn.Module):
    def __init__(self, D, mcfg):
        super().__init__()
        self.norm = RMSNorm(D)
        self.mamba = MambaBlock(mcfg)
        self.proj_w = nn.Linear(D, D, bias=False)
        self.proj_r = nn.Linear(D, D, bias=False)
        self.decay = nn.Parameter(torch.tensor(4.6))

    def forward(self, x, W=None):
        out = self.mamba(self.norm(x))
        B, T, D = out.shape
        lg = torch.sigmoid(self.decay).log()

        v = self.proj_w(out)
        wk = F.pad(out[:, :-1], (0, 0, 1, 0))
        pos = torch.arange(T, device=x.device)

        M = torch.exp((pos[:, None] - 1 - pos[None, :]).clamp(min=0) * lg)
        M = M * (pos[:, None] > pos[None, :])

        reads = torch.bmm(torch.bmm(out, wk.transpose(-1, -2)) * M, v)
        if W is not None:
            carry = torch.einsum("bij,btj->bti", W, out)
            reads = reads + carry * torch.exp(pos * lg)[None, :, None]

        w = torch.exp(torch.arange(T - 1, -1, -1, device=x.device) * lg)
        W_new = torch.einsum("t,btd,bte->bde", w, v, wk)
        if W is not None:
            W_new = W_new + torch.exp(torch.tensor(T, device=x.device) * lg) * W

        return x + out + 0.03 * self.proj_r(reads), W_new


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mcfg = MambaCfg(d_model=cfg.d_model, n_layers=cfg.n_layers,
                        d_state=cfg.d_state, d_conv=cfg.d_conv, expand_factor=cfg.expand)
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Layer(cfg.d_model, mcfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.emb.weight = self.head.weight

    def forward(self, ids, memories=None):
        x = self.emb(ids)
        mems = []
        for i, layer in enumerate(self.layers):
            x, m = layer(x, W=memories[i] if memories else None)
            mems.append(m)
        return self.head(self.norm(x)), mems


# -- Data --

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


def run_model(model, x, pairs):
    mid = 2 * pairs
    l1, mems = model(x[:, :mid])
    l2, _ = model(x[:, mid:], memories=mems)
    return torch.cat([l1, l2], dim=1)[:, :-1]


# -- Train + Eval --

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    cfg = Config(vocab_size=VOCAB, d_model=128, d_state=16, d_conv=4, expand=2, n_layers=4)
    model = Model(cfg).to(device)
    amp_ctx = torch.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"device={device}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  train_pairs={TRAIN_PAIRS}")

    # train
    for step in range(500):
        x, tgt, mask = make_batch(64, TRAIN_PAIRS, device)
        with amp_ctx:
            logits = run_model(model, x, TRAIN_PAIRS)

        loss = (F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), reduction="none")
                .view_as(mask).mul(mask).sum() / mask.sum())
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step == 499:
            with torch.no_grad():
                acc = ((logits.argmax(-1) == tgt) * mask).sum() / mask.sum()
            print(f"  step {step:3d}  loss={loss.item():.4f}  recall={acc.item():.1%}")

    # eval sweep
    eval_pairs = [4, 8, 16, 32, 64, 128, 256]
    results = {}
    model.eval()
    print("\ncapacity sweep:")
    with torch.no_grad(), amp_ctx:
        for p in eval_pairs:
            accs = []
            for _ in range(20):
                x, tgt, mask = make_batch(64, p, device)
                logits = run_model(model, x, p)
                accs.append(((logits.argmax(-1) == tgt) * mask).sum() / mask.sum())
            acc = torch.stack(accs).mean().item()
            results[p] = acc
            print(f"  {p:3d} pairs: {acc:.1%}")

    # plot
    pairs_list = list(results.keys())
    acc_list = [results[p] * 100 for p in pairs_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pairs_list, acc_list, "o-", linewidth=2, markersize=8)
    ax.axhline(y=100 / N_VALS, color="r", linestyle="--", alpha=0.5, label=f"chance ({100/N_VALS:.1f}%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(pairs_list)
    ax.set_xticklabels(pairs_list)
    ax.set_xlabel("Number of KV pairs")
    ax.set_ylabel("Recall accuracy (%)")
    ax.set_title(f"Hebbian memory capacity (d_model={cfg.d_model})")
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("capacity.png", dpi=150)
    print(f"\nsaved capacity.png")
