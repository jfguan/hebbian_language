"""Synthetic recall: can Hebbian memory carry associations across a Mamba reset?

Sequence: [k0 v0 k1 v1 ... | kπ(0) vπ(0) kπ(1) vπ(1) ...]
           store phase        query phase (Mamba reset, only W persists)

No filler. Loss only on recall positions.
Without memory: chance = 1/16. With memory: should reach ~100%.

Uses a self-contained model with chunk support (forward passes W across splits).
The main model.py no longer supports chunks since full-sequence training won.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import MambaBlock, MambaConfig as MambaCfg, RMSNorm
from model import Config

NUM_KEYS = 16
NUM_VALS = 16
VOCAB = NUM_KEYS + NUM_VALS
PAIRS = 8
SEQ_LEN = 4 * PAIRS
MID = SEQ_LEN // 2


class RecallLayer(nn.Module):
    """HebbianMambaLayer with chunk support (carries W across forward calls)."""

    def __init__(self, cfg, mcfg):
        super().__init__()
        D = cfg.d_model
        self.d_model = D
        self.use_memory = cfg.use_memory
        self.norm = RMSNorm(D)
        self.mamba = MambaBlock(mcfg)
        if self.use_memory:
            self.proj_write = nn.Linear(D, D, bias=False)
            self.proj_read = nn.Linear(D, D, bias=False)
            self.decay = nn.Parameter(torch.tensor(4.6))

    def _memory_attend(self, out, W):
        B, T, D = out.shape
        log_gamma = torch.sigmoid(self.decay).log()
        v = self.proj_write(out)
        wk = F.pad(out[:, :-1], (0, 0, 1, 0))
        rk = out
        pos = torch.arange(T, device=out.device)
        diffs = (pos[:, None] - 1 - pos[None, :]).clamp(min=0)
        M = torch.exp(diffs * log_gamma) * (pos[:, None] > pos[None, :])
        reads = torch.bmm(torch.bmm(rk, wk.transpose(-1, -2)) * M, v)
        if W is not None:
            carry = torch.einsum("bij,btj->bti", W, rk)
            reads = reads + carry * torch.exp(pos * log_gamma)[None, :, None]
        out = out + 0.03 * self.proj_read(reads)
        w = torch.exp(torch.arange(T - 1, -1, -1, device=out.device) * log_gamma)
        W_new = torch.einsum("t,btd,bte->bde", w, v, wk)
        if W is not None:
            W_new = W_new + torch.exp(T * log_gamma) * W
        return out, W_new

    def forward(self, x, memory=None):
        residual = x
        out = self.mamba(self.norm(x))
        if self.use_memory:
            out, memory = self._memory_attend(out, memory)
        return residual + out, memory


class RecallModel(nn.Module):
    """Small Hebbian Mamba with chunk support for the recall test."""

    def __init__(self, cfg):
        super().__init__()
        mcfg = MambaCfg(
            d_model=cfg.d_model, n_layers=cfg.n_layers,
            d_state=cfg.d_state, d_conv=cfg.d_conv, expand_factor=cfg.expand,
        )
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([RecallLayer(cfg, mcfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight

    def forward(self, input_ids, memories=None):
        x = self.embedding(input_ids)
        new_mems = []
        for i, layer in enumerate(self.layers):
            x, mem = layer(x, memory=memories[i] if memories else None)
            new_mems.append(mem)
        logits = self.lm_head(self.norm(x))
        return logits, new_mems


def make_batch(B, device):
    x = torch.zeros(B, SEQ_LEN, dtype=torch.long, device=device)

    for b in range(B):
        keys = torch.randperm(NUM_KEYS, device=device)[:PAIRS]
        vals = torch.randint(NUM_KEYS, VOCAB, (PAIRS,), device=device)

        for i in range(PAIRS):
            x[b, 2 * i] = keys[i]
            x[b, 2 * i + 1] = vals[i]

        perm = torch.randperm(PAIRS, device=device)
        for i in range(PAIRS):
            x[b, MID + 2 * i] = keys[perm[i]]
            x[b, MID + 2 * i + 1] = vals[perm[i]]

    mask = torch.zeros(B, SEQ_LEN - 1, device=device)
    for i in range(PAIRS):
        mask[:, MID + 2 * i] = 1

    targets = x[:, 1:]
    return x, targets, mask


def train_and_eval(use_memory, device, steps=250, B=64, lr=1e-3):
    cfg = Config(
        vocab_size=VOCAB, d_model=128, d_state=16,
        d_conv=4, expand=2, n_layers=4, use_memory=use_memory,
    )
    model = RecallModel(cfg).to(device)
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        x, targets, mask = make_batch(B, device)

        logits1, mems = model(x[:, :MID])
        logits2, _ = model(x[:, MID:], memories=mems)
        logits = torch.cat([logits1, logits2], dim=1)[:, :-1]

        per_tok = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1), reduction="none")
        per_tok = per_tok.view(B, SEQ_LEN - 1)
        loss = (per_tok * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                preds = logits.argmax(-1)
                correct = ((preds == targets) * mask).sum()
                acc = (correct / mask.sum()).item()
            print(f"  step {step:4d} | loss {loss.item():.4f} | recall {acc:.1%}")

    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(20):
            x, targets, mask = make_batch(B, device)
            logits1, mems = model(x[:, :MID])
            logits2, _ = model(x[:, MID:], memories=mems)
            logits = torch.cat([logits1, logits2], dim=1)[:, :-1]
            preds = logits.argmax(-1)
            correct = ((preds == targets) * mask).sum()
            accs.append((correct / mask.sum()).item())
    return sum(accs) / len(accs)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Task: recall {PAIRS} pairs after Mamba reset")
    print(f"Chance: {1/NUM_VALS:.1%}\n")

    torch.manual_seed(42)
    print("With memory:")
    acc_mem = train_and_eval(use_memory=True, device=device)

    torch.manual_seed(42)
    print("\nWithout memory:")
    acc_nomem = train_and_eval(use_memory=False, device=device)

    print(f"\nRecall accuracy:")
    print(f"  With memory:    {acc_mem:.1%}")
    print(f"  Without memory: {acc_nomem:.1%}")
    print(f"  Chance:         {1/NUM_VALS:.1%}")


if __name__ == "__main__":
    main()
