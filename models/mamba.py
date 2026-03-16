"""Pure Mamba (no Hebbian memory)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.mamba import MambaBlock, MambaConfig as MambaCfg, RMSNorm

from train.configs import ModelConfig


class MambaLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, mcfg: MambaCfg):
        super().__init__()
        self.d_inner = mcfg.d_inner
        self.d_conv = mcfg.d_conv
        self.norm = RMSNorm(cfg.d_model)
        self.mamba = MambaBlock(mcfg)

    def forward(self, x):
        return x + self.mamba(self.norm(x))

    def step(self, x, state=None):
        B = x.shape[0]
        if state is None:
            cache = (None, x.new_zeros(B, self.d_inner, self.d_conv - 1))
        else:
            cache = state
        out, cache = self.mamba.step(self.norm(x), cache)
        return x + out, cache


class Mamba(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        mcfg = MambaCfg(
            d_model=cfg.d_model, n_layers=cfg.n_layers,
            d_state=cfg.d_state, d_conv=cfg.d_conv, expand_factor=cfg.expand,
        )
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [MambaLayer(cfg, mcfg) for _ in range(cfg.n_layers)]
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
