"""Hybrid Hebbian + Delta Hebbian model.

Configurable layer pattern: plain Hebbian for statistical accumulation,
delta Hebbian for precise recall, conv-only for layers that don't need memory.
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
    """Conv local mixing + optional memory (plain, delta, or none) + residual."""

    def __init__(self, cfg: ModelConfig, mode: str = "hebbian"):
        """mode: 'hebbian', 'delta', or 'conv_only'."""
        super().__init__()
        d_inner = cfg.expand * cfg.d_model
        self.norm = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg.d_model, expand=cfg.expand)
        self.conv = CausalConv(d_inner, d_conv=cfg.d_conv)

        if mode == "delta":
            self.memory = DeltaHebbianBlock(
                d_model=cfg.d_model,
                head_dim=cfg.head_dim or cfg.d_model,
                chunk_size=cfg.chunk_size,
                memory_alpha=cfg.memory_alpha,
                neg_eigenvalues=cfg.neg_eigenvalues,
            )
        elif mode == "hebbian":
            self.memory = HebbianBlock(
                d_model=cfg.d_model,
                chunk_size=cfg.chunk_size,
                memory_alpha=cfg.memory_alpha,
            )
        else:
            self.memory = None

    def forward(self, x):
        normed = self.norm(x)
        val = self.conv(self.mlp.project_up(normed))
        out = self.mlp(normed, val)
        if self.memory is not None:
            out = self.memory(out)
        return x + out

    def step(self, x, state=None):
        conv_st = state["conv"] if state else None
        mem_state = state["memory"] if state else None

        normed = self.norm(x)
        val, conv_st = self.conv.step(self.mlp.project_up(normed), conv_st)
        out = self.mlp(normed, val)
        if self.memory is not None:
            out, mem_state = self.memory.step(out, mem_state)

        return x + out, {"conv": conv_st, "memory": mem_state}


def _parse_layers(s):
    """Parse comma-separated layer indices, e.g. '3,7' -> {3, 7}."""
    if not s:
        return set()
    return {int(x) for x in s.split(",")}


class DeltaHebbianConv(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        delta_set = _parse_layers(cfg.delta_layers)
        no_mem_set = _parse_layers(cfg.no_memory_layers)

        layers = []
        for i in range(cfg.n_layers):
            if i in no_mem_set:
                mode = "conv_only"
            elif i in delta_set:
                mode = "delta"
            else:
                mode = "hebbian"
            layers.append(HybridHebbianLayer(cfg, mode=mode))

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(layers)
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
