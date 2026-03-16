"""Hybrid Hebbian + Delta Hebbian model.

3:1 ratio: plain Hebbian layers for statistical accumulation,
delta Hebbian every 4th layer for precise recall.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hebbian_components import CausalConv, GatedMLP, HebbianBlock, DeltaHebbianBlock
from train.configs import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class HybridHebbianLayer(nn.Module):
    """Conv local mixing + memory (plain or delta) + residual."""

    def __init__(self, cfg: ModelConfig, use_delta: bool = False):
        super().__init__()
        d_inner = cfg.expand * cfg.d_model
        self.norm = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg.d_model, expand=cfg.expand)
        self.conv = CausalConv(d_inner, d_conv=cfg.d_conv)
        if use_delta:
            self.memory = DeltaHebbianBlock(
                d_model=cfg.d_model,
                head_dim=cfg.d_model,
                chunk_size=cfg.chunk_size,
                memory_alpha=cfg.memory_alpha,
            )
        else:
            self.memory = HebbianBlock(
                d_model=cfg.d_model,
                chunk_size=cfg.chunk_size,
                memory_alpha=cfg.memory_alpha,
            )

    def forward(self, x):
        normed = self.norm(x)
        val = self.conv(self.mlp.project_up(normed))
        out = self.mlp(normed, val)
        out = self.memory(out)
        return x + out

    def step(self, x, state=None):
        conv_st = state["conv"] if state else None
        mem_state = state["memory"] if state else None

        normed = self.norm(x)
        val, conv_st = self.conv.step(self.mlp.project_up(normed), conv_st)
        out = self.mlp(normed, val)
        out, mem_state = self.memory.step(out, mem_state)

        return x + out, {"conv": conv_st, "memory": mem_state}


class DeltaHebbianConv(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([
            HybridHebbianLayer(cfg, use_delta=(i % 4 == 3))
            for i in range(cfg.n_layers)
        ])
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def step(self, token, states=None):
        x = self.embedding(token).squeeze(1)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer.step(x, state=states[i] if states else None)
            new_states.append(s)
        return self.lm_head(self.norm(x)), new_states
