"""Conv + Hebbian associative memory for FineWeb benchmark.

Target: GPT-2 124M scale on FineWeb (3.28 val loss target).
Config: D=768, L=18, expand=2, d_conv=4, vocab=50304 → ~124M params.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50304  # GPT-2 (50257 padded to multiple of 64)
    d_model: int = 768
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 18
    memory_alpha: float = 0.03
    chunk_size: int = 64


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return (
            x
            * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(
                x.dtype
            )
            * self.weight
        )


class HebbianLayer(nn.Module):
    """Conv local mixing + Hebbian associative memory + residual."""

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        E = cfg.expand
        d_inner = E * D
        self.d_model = D
        self.d_inner = d_inner
        self.d_conv = cfg.d_conv
        self.chunk_size = cfg.chunk_size

        # Conv block
        self.norm = RMSNorm(D)
        self.proj = nn.Linear(D, d_inner, bias=False)
        self.gate = nn.Linear(D, d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            cfg.d_conv,
            bias=True,
            groups=d_inner,
            padding=cfg.d_conv - 1,
        )
        self.out_proj = nn.Linear(d_inner, D, bias=False)

        # Hebbian memory
        self.proj_write = nn.Linear(D, D, bias=False)
        self.proj_read = nn.Linear(D, D, bias=False)
        self.decay = nn.Parameter(torch.tensor(4.6))  # σ(4.6) ≈ 0.99
        self.log_alpha = nn.Parameter(torch.tensor(cfg.memory_alpha).log())

    def _conv(self, x):
        B, L, D = x.shape
        val = self.proj(x).transpose(1, 2)
        val = F.silu(self.conv1d(val)[:, :, :L].transpose(1, 2))
        return self.out_proj(val * F.silu(self.gate(x)))

    def _memory_attend(self, out):
        B, T, D = out.shape
        C = self.chunk_size
        out32 = out.float()

        gamma = torch.sigmoid(self.decay)
        log_gamma = gamma.log()

        v = self.proj_write(out32)
        wk = F.pad(out32[:, :-1], (0, 0, 1, 0))
        rk = out32

        W = out32.new_zeros(B, D, D)
        reads_list = []

        for start in range(0, T, C):
            end = min(start + C, T)
            Ci = end - start
            p = torch.arange(Ci, device=out.device)

            rk_c, wk_c, v_c = rk[:, start:end], wk[:, start:end], v[:, start:end]

            inter = torch.matmul(W, rk_c.transpose(1, 2)).transpose(1, 2)
            inter = inter * (gamma**p)[None, :, None]

            S = torch.bmm(rk_c, wk_c.transpose(1, 2))
            diffs = (p[:, None] - 1 - p[None, :]).clamp(min=0)
            M = torch.exp(diffs * log_gamma) * (p[:, None] > p[None, :])
            intra = torch.bmm(S * M, v_c)

            reads_list.append(inter + intra)

            gw = (gamma ** (Ci - 1 - p))[None, :, None]
            W = gamma**Ci * W + torch.bmm((v_c * gw).transpose(1, 2), wk_c)

        reads = torch.cat(reads_list, dim=1)
        return out + self.log_alpha.exp() * self.proj_read(reads).to(out.dtype)

    def forward(self, x):
        out = self._conv(self.norm(x))
        out = self._memory_attend(out)
        return x + out

    def step(self, x, state=None):
        B, D = x.shape
        if state is None:
            conv_st = x.new_zeros(B, self.d_inner, self.d_conv - 1)
            W = x.new_zeros(B, D, D)
            r_prev = x.new_zeros(B, D)
        else:
            conv_st, W, r_prev = state

        normed = self.norm(x)
        val = self.proj(normed)
        conv_input = torch.cat([conv_st, val.unsqueeze(-1)], dim=-1)
        conv_st = conv_input[:, :, 1:]
        val = F.silu(
            (conv_input * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias
        )
        out = self.out_proj(val * F.silu(self.gate(normed)))
        raw_out = out  # save before memory augmentation for next step's write key

        gamma = torch.sigmoid(self.decay)
        read = torch.einsum("bij,bj->bi", W, out)
        W = gamma * W + torch.einsum("bi,bj->bij", self.proj_write(out), r_prev)
        out = out + self.log_alpha.exp() * self.proj_read(read)

        return x + out, (conv_st, W, raw_out)


class HebbianConv(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [HebbianLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight  # weight tying

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def step(self, token, states=None):
        x = self.embedding(token).squeeze(1)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, s = layer.step(x, state=states[i] if states else None)
            new_states.append(s)
        return self.lm_head(self.norm(x)), new_states
