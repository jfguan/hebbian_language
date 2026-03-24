"""Dual-matrix delta Hebbian memory.

Each layer has two D×D state matrices with shared keys (one WY) but
separate values (proj_write D→2D, split). Double the value capacity
per association at D² extra params per layer.

Layer structure: conv + MLP + DualDeltaBlock + residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hebbian_components import CausalConv, GatedMLP
from train.configs import ModelConfig


# -- components --

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class DualDeltaBlock(nn.Module):
    """Two D×D delta memories with shared keys, separate values.

    proj_write maps D → 2D, split into v1 and v2.
    Shared token-shifted keys → one WY correction for both.
    Reads summed: (W1 @ rk + W2 @ rk) → out_proj.
    """

    def __init__(self, d_model: int, num_heads: int = 8, chunk_size: int = 64):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.n_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size

        # D → 2D: split into v1, v2
        self.proj_write = nn.Linear(d_model, 2 * d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, num_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # write gates: separate per matrix
        self.beta1_proj = nn.Linear(d_model, num_heads, bias=False)
        self.beta2_proj = nn.Linear(d_model, num_heads, bias=False)

        # decay: separate per matrix (different timescales)
        self.alpha1_proj = nn.Linear(d_model, num_heads, bias=False)
        self.alpha2_proj = nn.Linear(d_model, num_heads, bias=False)
        dt1 = torch.empty(num_heads).uniform_(0, 1) * (torch.tensor(0.1).log() - torch.tensor(0.001).log()) + torch.tensor(0.001).log()
        dt2 = torch.empty(num_heads).uniform_(0, 1) * (torch.tensor(0.1).log() - torch.tensor(0.001).log()) + torch.tensor(0.001).log()
        self.dt_bias1 = nn.Parameter(dt1.exp() + torch.log(-torch.expm1(-dt1.exp())))
        self.dt_bias2 = nn.Parameter(dt2.exp() + torch.log(-torch.expm1(-dt2.exp())))
        self.dt_bias1._no_weight_decay = True
        self.dt_bias2._no_weight_decay = True
        self.A_log1 = nn.Parameter(torch.empty(num_heads).uniform_(0, 4).log())
        self.A_log2 = nn.Parameter(torch.empty(num_heads).uniform_(0, 4).log())
        self.A_log1._no_weight_decay = True
        self.A_log2._no_weight_decay = True

        # static masks
        C = chunk_size
        self.register_buffer("causal_mask", torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0), persistent=False)
        self.register_buffer("eye_C", torch.eye(C), persistent=False)

    def forward(self, out):
        """out: (B, T, D). returns: (B, T, D)."""
        B, T, D = out.shape
        H, d, C = self.n_heads, self.head_dim, self.chunk_size
        x = out.float()

        # -- projections --
        vals = self.proj_write(x).view(B, T, 2, H, d)
        v1, v2 = vals[:, :, 0], vals[:, :, 1]  # each (B, T, H, d)
        gate = self.gate_proj(x).sigmoid().view(B, T, H, 1)

        beta1 = self.beta1_proj(x).sigmoid()
        beta2 = self.beta2_proj(x).sigmoid()
        decay1 = -self.A_log1.exp().view(1, 1, H) * F.softplus(self.alpha1_proj(x) + self.dt_bias1)
        decay2 = -self.A_log2.exp().view(1, 1, H) * F.softplus(self.alpha2_proj(x) + self.dt_bias2)

        # normalized keys with token shift
        rk = F.normalize(x.view(B, T, H, d), dim=-1)
        wk = F.pad(rk[:, :-1], (0, 0, 0, 0, 1, 0))

        # transpose to (B, H, T, d)
        rk, wk, v1, v2 = [t.transpose(1, 2).float() for t in (rk, wk, v1, v2)]

        # scale by beta
        v1 = v1 * beta1.transpose(1, 2).unsqueeze(-1)
        v2 = v2 * beta2.transpose(1, 2).unsqueeze(-1)
        wk_beta1 = wk * beta1.transpose(1, 2).unsqueeze(-1)
        wk_beta2 = wk * beta2.transpose(1, 2).unsqueeze(-1)

        # -- pad to chunk boundary --
        pad = (C - (T % C)) % C
        if pad > 0:
            rk = F.pad(rk, (0, 0, 0, pad))
            wk = F.pad(wk, (0, 0, 0, pad))
            v1 = F.pad(v1, (0, 0, 0, pad))
            v2 = F.pad(v2, (0, 0, 0, pad))
            wk_beta1 = F.pad(wk_beta1, (0, 0, 0, pad))
            wk_beta2 = F.pad(wk_beta2, (0, 0, 0, pad))
            decay1 = F.pad(decay1, (0, 0, 0, pad))
            decay2 = F.pad(decay2, (0, 0, 0, pad))

        T_pad = rk.shape[2]
        N = T_pad // C

        # -- reshape into chunks --
        rk = rk.view(B, H, N, C, d)
        wk = wk.view(B, H, N, C, d)
        v1 = v1.view(B, H, N, C, d)
        v2 = v2.view(B, H, N, C, d)
        wk_beta1 = wk_beta1.view(B, H, N, C, d)
        wk_beta2 = wk_beta2.view(B, H, N, C, d)
        decay1 = decay1.transpose(1, 2).view(B, H, N, C)
        decay2 = decay2.transpose(1, 2).view(B, H, N, C)

        # -- shared decay mask for WY (using decay1 for keys) --
        # WY correction depends on key interactions, not values
        # Use average decay for the shared mask
        decay_avg = (decay1 + decay2) * 0.5
        cum_decay = decay_avg.cumsum(-1)
        L_mask = (cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)).tril().exp().tril()

        # -- one WY correction (shared keys) --
        # Use wk_beta1 for WY since keys are shared; beta difference is in values
        wk_beta_avg = (wk_beta1 + wk_beta2) * 0.5
        A = -(wk_beta_avg @ wk.transpose(-1, -2) * L_mask).masked_fill(self.causal_mask, 0)
        A = A.clone()
        for i in range(1, C):
            A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :i].clone().unsqueeze(-1) * A[..., :i, :i].clone()).sum(-2)
        A = A + self.eye_C

        # -- per-matrix decay masks --
        cum1 = decay1.cumsum(-1)
        cum2 = decay2.cumsum(-1)
        decay_exp1 = cum1.unsqueeze(-1).exp()
        decay_exp2 = cum2.unsqueeze(-1).exp()
        L1 = (cum1.unsqueeze(-1) - cum1.unsqueeze(-2)).tril().exp().tril()
        L2 = (cum2.unsqueeze(-1) - cum2.unsqueeze(-2)).tril().exp().tril()

        # -- corrected values --
        v1_corr = A @ v1
        v2_corr = A @ v2
        wk_cumdecay1 = A @ (wk_beta1 * decay_exp1)
        wk_cumdecay2 = A @ (wk_beta2 * decay_exp2)

        # -- intra-chunk attention (shared keys, per-matrix decay) --
        rk_wk = rk @ wk.transpose(-1, -2)
        intra1 = (rk_wk * L1).masked_fill(torch.triu(torch.ones(C, C, device=x.device, dtype=torch.bool), diagonal=1), 0)
        intra2 = (rk_wk * L2).masked_fill(torch.triu(torch.ones(C, C, device=x.device, dtype=torch.bool), diagonal=1), 0)

        # -- chunk-by-chunk propagation (two state matrices) --
        S1 = x.new_zeros(B, H, d, d)
        S2 = x.new_zeros(B, H, d, d)
        o = x.new_zeros(B, H, N, C, d)

        for i in range(N):
            rk_i, wk_i = rk[:, :, i], wk[:, :, i]

            # matrix 1
            v1_new = v1_corr[:, :, i] - wk_cumdecay1[:, :, i] @ S1
            o1 = (rk_i * decay_exp1[:, :, i]).unsqueeze(-2) @ S1.unsqueeze(-3)
            o1 = o1.squeeze(-2) + intra1[:, :, i] @ v1_new
            dw1 = (cum1[:, :, i, -1, None] - cum1[:, :, i]).exp().unsqueeze(-1)
            S1 = S1 * cum1[:, :, i, -1, None, None].exp() + (wk_i * dw1).transpose(-1, -2) @ v1_new
            S1_norm = S1.norm(dim=(-2, -1), keepdim=True)
            S1 = S1 * (S1_norm.clamp(max=100.0) / S1_norm.clamp(min=1e-6))

            # matrix 2
            v2_new = v2_corr[:, :, i] - wk_cumdecay2[:, :, i] @ S2
            o2 = (rk_i * decay_exp2[:, :, i]).unsqueeze(-2) @ S2.unsqueeze(-3)
            o2 = o2.squeeze(-2) + intra2[:, :, i] @ v2_new
            dw2 = (cum2[:, :, i, -1, None] - cum2[:, :, i]).exp().unsqueeze(-1)
            S2 = S2 * cum2[:, :, i, -1, None, None].exp() + (wk_i * dw2).transpose(-1, -2) @ v2_new
            S2_norm = S2.norm(dim=(-2, -1), keepdim=True)
            S2 = S2 * (S2_norm.clamp(max=100.0) / S2_norm.clamp(min=1e-6))

            o[:, :, i] = o1 + o2

        # -- output --
        o = o.view(B, H, T_pad, d).transpose(1, 2)[:, :T]
        o = o * gate
        return out + self.out_proj(o.reshape(B, T, D)).to(out.dtype)

    def step(self, out, state=None):
        """Sequential recurrence. out: (B, D). returns: (B, D), state."""
        B, D = out.shape
        H, d = self.n_heads, self.head_dim

        vals = self.proj_write(out).view(B, H, 2, d)
        v1, v2 = vals[:, :, 0], vals[:, :, 1]
        gate = self.gate_proj(out).sigmoid().view(B, H, 1)
        beta1 = self.beta1_proj(out).sigmoid().unsqueeze(-1)
        beta2 = self.beta2_proj(out).sigmoid().unsqueeze(-1)
        decay1 = (-self.A_log1.exp() * F.softplus(self.alpha1_proj(out) + self.dt_bias1)).exp().view(B, H, 1, 1)
        decay2 = (-self.A_log2.exp() * F.softplus(self.alpha2_proj(out) + self.dt_bias2)).exp().view(B, H, 1, 1)

        rk = F.normalize(out.view(B, H, d), dim=-1)

        if state is not None:
            W1, W2, wk = state["W1"], state["W2"], state["wk"]
        else:
            W1 = out.new_zeros(B, H, d, d)
            W2 = out.new_zeros(B, H, d, d)
            wk = out.new_zeros(B, H, d)

        # decay
        W1 = W1 * decay1
        W2 = W2 * decay2

        # delta write
        err1 = (v1 - (W1 * wk.unsqueeze(-1)).sum(-2)) * beta1
        err2 = (v2 - (W2 * wk.unsqueeze(-1)).sum(-2)) * beta2
        W1 = W1 + wk.unsqueeze(-1) * err1.unsqueeze(-2)
        W2 = W2 + wk.unsqueeze(-1) * err2.unsqueeze(-2)

        # read
        read = (W1 * rk.unsqueeze(-1)).sum(-2) + (W2 * rk.unsqueeze(-1)).sum(-2)
        read = read * gate
        read = self.out_proj(read.reshape(B, D))

        return out + read, {"W1": W1, "W2": W2, "wk": rk}


# -- layer and model --

class DualDeltaLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d_inner = cfg.expand * cfg.d_model
        self.norm = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg.d_model, expand=cfg.expand)
        self.conv = CausalConv(d_inner, d_conv=cfg.d_conv)
        self.memory = DualDeltaBlock(
            d_model=cfg.d_model,
            num_heads=cfg.delta_num_heads or 8,
            chunk_size=cfg.chunk_size,
        )

    def forward(self, x):
        normed = self.norm(x)
        val = self.conv(self.mlp.project_up(normed))
        out = self.mlp(normed, val)
        out = self.memory(out)
        return x + out

    def step(self, x, state=None):
        conv_st = state["conv"] if state else None
        mem_st = state["memory"] if state else None
        normed = self.norm(x)
        val, conv_st = self.conv.step(self.mlp.project_up(normed), conv_st)
        out = self.mlp(normed, val)
        out, mem_st = self.memory.step(out, mem_st)
        return x + out, {"conv": conv_st, "memory": mem_st}


class DualDelta(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        from models.swa_delta import SWALayer
        delta_set = set(cfg.delta_layers or [])
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([
            DualDeltaLayer(cfg) if i in delta_set else SWALayer(cfg)
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
