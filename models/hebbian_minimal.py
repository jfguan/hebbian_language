"""Conv + Hebbian associative memory.

Depthwise conv for local feature extraction (kernel=d_conv tokens),
Hebbian outer-product memory for long-range associative binding.

Memory: W_t = γW_{t-1} + v_t⊗k_t, read_t = W_{t-1}·rk_t
Training uses O(TC·D + T·D²) chunkwise parallel form; inference uses O(D²) recurrent form.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hebbian_components import CausalConv, GatedMLP, HebbianBlock


@dataclass
class Config:
    vocab_size: int = 384
    d_model: int = 512
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 8
    memory_alpha: float = 0.03
    chunk_size: int = 64


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class HebbianLayer(nn.Module):
    """Conv local mixing + Hebbian associative memory + residual."""

    def __init__(self, cfg: Config):
        super().__init__()
        d_inner = cfg.expand * cfg.d_model
        self.norm = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg.d_model, expand=cfg.expand)
        self.conv = CausalConv(d_inner, d_conv=cfg.d_conv)
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


class HebbianConv(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([HebbianLayer(cfg) for _ in range(cfg.n_layers)])
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
