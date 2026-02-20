"""Mamba + Hebbian associative memory.

Memory: W_t = γW_{t-1} + v_t⊗k_t, read_t = W_t·k_t
Training uses O(T²D) parallel form; inference uses O(D²) recurrent form.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import MambaBlock, MambaConfig as MambaCfg, RMSNorm


@dataclass
class Config:
    vocab_size: int = 384
    d_model: int = 512
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 8
    use_memory: bool = True
    dual_memory: bool = False  # second W matrix with slower decay


class HebbianMambaLayer(nn.Module):
    def __init__(self, cfg: Config, mcfg: MambaCfg):
        super().__init__()
        D = cfg.d_model
        self.d_model = D
        self.d_inner = mcfg.d_inner
        self.d_conv = mcfg.d_conv
        self.use_memory = cfg.use_memory

        self.norm = RMSNorm(D)
        self.mamba = MambaBlock(mcfg)

        self.dual_memory = cfg.dual_memory
        if self.use_memory:
            self.proj_write = nn.Linear(D, D, bias=False)
            self.proj_read = nn.Linear(D, D, bias=False)
            self.decay = nn.Parameter(torch.tensor(4.6))   # σ(4.6) ≈ 0.99
            if self.dual_memory:
                self.decay_slow = nn.Parameter(torch.tensor(6.9))  # σ(6.9) ≈ 0.999

    def _memory_attend(self, out):
        """Parallel form: reads = (M ⊙ (rk·wk^T)) · v where M is causal decay mask."""
        B, T, D = out.shape
        # Upcast to float32 — bf16 backward overflows without scaling
        out32 = out.float()
        log_gamma = torch.sigmoid(self.decay).log()

        v = self.proj_write(out32)                      # write values
        wk = F.pad(out32[:, :-1], (0, 0, 1, 0))        # write keys (shifted)
        rk = out32                                       # read keys (current)

        # Causal decay: M[t,s] = γ^(t-1-s) · 𝟙[s<t]
        pos = torch.arange(T, device=out.device)
        diffs = (pos[:, None] - 1 - pos[None, :]).clamp(min=0)
        causal = (pos[:, None] > pos[None, :])
        M = torch.exp(diffs * log_gamma) * causal

        scores = torch.bmm(rk, wk.transpose(-1, -2))
        reads = torch.bmm(scores * M, v)
        if self.dual_memory:
            log_gamma_slow = torch.sigmoid(self.decay_slow).log()
            M_slow = torch.exp(diffs * log_gamma_slow) * causal
            reads = reads + torch.bmm(scores * M_slow, v)
        return out + 0.03 * self.proj_read(reads).to(out.dtype)

    def forward(self, x):
        residual = x
        out = self.mamba(self.norm(x))
        if self.use_memory:
            out = self._memory_attend(out)
        return residual + out

    def step(self, x, state=None):
        """Recurrent form: W ← γW + v⊗wk, read = W·out."""
        B = x.shape[0]
        residual = x

        if state is None:
            cache = (None, x.new_zeros(B, self.d_inner, self.d_conv - 1))
        else:
            cache = state["cache"]

        out, cache = self.mamba.step(self.norm(x), cache)

        if not self.use_memory:
            return residual + out, {"cache": cache}

        W = state["memory"] if state else x.new_zeros(B, self.d_model, self.d_model)
        r_prev = state["r_prev"] if state else x.new_zeros(B, self.d_model)

        gamma = torch.sigmoid(self.decay)
        raw_out = out
        write = torch.einsum("bi,bj->bij", self.proj_write(out), r_prev)
        read = torch.einsum("bij,bj->bi", W, out)
        W = gamma * W + write

        if self.dual_memory:
            W_slow = state["memory_slow"] if state else x.new_zeros(B, self.d_model, self.d_model)
            gamma_slow = torch.sigmoid(self.decay_slow)
            read = read + torch.einsum("bij,bj->bi", W_slow, out)
            W_slow = gamma_slow * W_slow + write

        out = out + 0.03 * self.proj_read(read)

        new_state = {"cache": cache, "memory": W, "r_prev": raw_out}
        if self.dual_memory:
            new_state["memory_slow"] = W_slow
        return residual + out, new_state


class HebbianMamba(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        mcfg = MambaCfg(
            d_model=cfg.d_model, n_layers=cfg.n_layers,
            d_state=cfg.d_state, d_conv=cfg.d_conv, expand_factor=cfg.expand,
        )
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [HebbianMambaLayer(cfg, mcfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def step(self, token, states=None):
        x = self.embedding(token)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer.step(x, state=states[i] if states else None)
            new_states.append(s)
        return self.lm_head(self.norm(x)), new_states
