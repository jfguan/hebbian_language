"""Conv + MLP only baseline (no memory).

Same architecture as Hebbian model but without the HebbianBlock.
Pure local mixing via depthwise causal conv + SwiGLU MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hebbian_components import CausalConv, GatedMLP
from train.configs import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class ConvLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d_inner = cfg.expand * cfg.d_model
        self.norm = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg.d_model, expand=cfg.expand)
        self.conv = CausalConv(d_inner, d_conv=cfg.d_conv)

    def forward(self, x):
        normed = self.norm(x)
        val = self.conv(self.mlp.project_up(normed))
        return x + self.mlp(normed, val)

    def step(self, x, state=None):
        normed = self.norm(x)
        val, state = self.conv.step(self.mlp.project_up(normed), state)
        return x + self.mlp(normed, val), state


class ConvOnly(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([ConvLayer(cfg) for _ in range(cfg.n_layers)])
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
