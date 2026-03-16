"""Mamba + Hebbian associative memory.

Memory: W_t = γW_{t-1} + v_t⊗k_t, read_t = W_{t-1}·rk_t
Training uses O(TC·D + T·D²) chunkwise parallel form; inference uses O(D²) recurrent form.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import MambaBlock, MambaConfig as MambaCfg, RMSNorm

from models.hebbian_components import DeltaHebbianBlock


@dataclass
class Config:
    vocab_size: int = 384
    d_model: int = 512
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 8
    memory_alpha: float = 0.03
    head_dim: int = 128


class HebbianMambaLayer(nn.Module):
    def __init__(self, cfg: Config, mcfg: MambaCfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.d_inner = mcfg.d_inner
        self.d_conv = mcfg.d_conv

        self.norm = RMSNorm(cfg.d_model)
        self.mamba = MambaBlock(mcfg)
        self.memory = DeltaHebbianBlock(
            d_model=cfg.d_model,
            head_dim=cfg.head_dim,
        )

    def forward(self, x):
        out = self.mamba(self.norm(x))
        out = self.memory(out)
        return x + out

    def step(self, x, state=None):
        B = x.shape[0]
        if state is None:
            cache = (None, x.new_zeros(B, self.d_inner, self.d_conv - 1))
        else:
            cache = state["cache"]

        out, cache = self.mamba.step(self.norm(x), cache)

        mem_state = state.get("memory") if state else None
        out, mem_state = self.memory.step(out, mem_state)

        return x + out, {"cache": cache, "memory": mem_state}


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
