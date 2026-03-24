"""Reusable components for Hebbian models.

- CausalConv: causal depthwise conv1d for local token mixing
- GatedMLP: SwiGLU gated projections for channel mixing
- HebbianBlock: simple outer-product associative memory
- DeltaHebbianBlock: multi-head delta rule memory with error correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv(nn.Module):
    """Causal depthwise conv1d with parallel and recurrent forms."""

    def __init__(self, d: int, d_conv: int = 4):
        super().__init__()
        self.d = d
        self.d_conv = d_conv
        self.conv1d = nn.Conv1d(d, d, d_conv, bias=True, groups=d, padding=d_conv - 1)

    def forward(self, x):
        """(B, L, D) -> (B, L, D)."""
        return self.conv1d(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)

    def step(self, x, state=None):
        """(B, D) -> (B, D), state."""
        if state is None:
            state = x.new_zeros(x.shape[0], self.d, self.d_conv - 1)
        conv_input = torch.cat([state, x.unsqueeze(-1)], dim=-1)
        state = conv_input[:, :, 1:]
        assert self.conv1d.bias is not None
        out = (conv_input * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias
        return out, state


class GatedMLP(nn.Module):
    """SwiGLU gated projections: up-project, gate, down-project."""

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.proj = nn.Linear(d_model, d_inner, bias=False)
        self.gate = nn.Linear(d_model, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def project_up(self, x):
        """(B, *, d_model) -> (B, *, d_inner)."""
        return self.proj(x)

    def forward(self, x, val):
        """Gate and project down. x: original input for gate, val: transformed value in d_inner space."""
        return self.out_proj(F.silu(val) * F.silu(self.gate(x)))


class SlidingWindowAttention(nn.Module):
    """Local attention with token-shifted keys. No Q/K projections.

    Query = current hidden state, Key = previous token's hidden state.
    Exact content-addressed retrieval within a fixed window.
    """

    def __init__(self, d_model: int, num_heads: int = 8, window_size: int = 256):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, num_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """x: (B, T, D). returns: (B, T, D)."""
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        # token shift: key is previous position, split into heads
        q = x.view(B, T, H, d).transpose(1, 2)                # (B, H, T, d)
        k = F.pad(x[:, :-1], (0, 0, 1, 0)).view(B, T, H, d).transpose(1, 2)
        v = self.proj_v(x).view(B, T, H, d).transpose(1, 2)
        gate = self.gate_proj(x).sigmoid().view(B, T, H, 1)   # (B, T, H, 1)

        # SDPA handles causal masking and flash attention kernel automatically
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, d)
        out = out.transpose(1, 2) * gate                       # (B, T, H, d)
        return self.out_proj(out.reshape(B, T, D))

    def step(self, x, state=None):
        """x: (B, D). returns: (B, D), new state."""
        B, D = x.shape
        W = self.window_size

        if state is None:
            state = {"keys": x.new_zeros(B, 0, D), "vals": x.new_zeros(B, 0, D)}

        # append current token as key and value for future positions
        k_prev = F.pad(state["keys"][:, -1:], (0, 0, 0, 0)) if state["keys"].shape[1] > 0 else x.new_zeros(B, 1, D)
        keys = torch.cat([state["keys"], k_prev], dim=1)[:, -W:]
        vals = torch.cat([state["vals"], self.proj_v(x).unsqueeze(1)], dim=1)[:, -W:]

        # attend: query is current x, keys are previous positions
        H, hd = self.num_heads, self.head_dim
        gate = self.gate_proj(x).sigmoid().view(B, H, 1)  # (B, H, 1)
        if keys.shape[1] > 0:
            scale = hd ** -0.5
            # reshape to multi-head for attention
            q = x.view(B, H, hd)                                      # (B, H, hd)
            k_mh = keys.view(B, -1, H, hd).transpose(1, 2)           # (B, H, K, hd)
            v_mh = vals.view(B, -1, H, hd).transpose(1, 2)           # (B, H, K, hd)
            scores = (q.unsqueeze(2) @ k_mh.transpose(-1, -2)).squeeze(2) * scale  # (B, H, K)
            attn = F.softmax(scores, dim=-1)
            out = (attn.unsqueeze(-1) * v_mh).sum(2) * gate           # (B, H, hd)
            out = self.out_proj(out.reshape(B, D))
        else:
            out = x.new_zeros(B, D)

        new_state = {"keys": torch.cat([state["keys"], x.unsqueeze(1)], dim=1)[:, -W:],
                     "vals": vals}
        return out, new_state


class HebbianBlock(nn.Module):
    """Associative memory with data-dependent decay, write gate, and output gate.

    W_t = e^g · W_{t-1} + β·v_t⊗k_{t-1}
    Token shift: write key is previous token, read key is current.
    No key normalization — raw hidden states as keys.
    """

    def __init__(self, d_model: int, chunk_size: int = 64, num_heads: int = 1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.n_heads = num_heads
        self.head_dim = d_model // num_heads

        H = self.n_heads
        self.proj_write = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, H, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # write gate
        self.beta_proj = nn.Linear(d_model, H, bias=False)

        # data-dependent decay (Mamba-style init)
        self.alpha_proj = nn.Linear(d_model, H, bias=False)
        dt = torch.empty(H).uniform_(0, 1) * (torch.tensor(0.1).log() - torch.tensor(0.001).log()) + torch.tensor(0.001).log()
        dt = dt.exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.A_log = nn.Parameter(torch.empty(H).uniform_(0, 4).log())
        self.A_log._no_weight_decay = True

    def forward(self, out):
        """Chunkwise parallel form with data-dependent decay.

        out: (B, T, D) hidden states.
        returns: (B, T, D) augmented with memory reads.
        """
        B, T, D = out.shape
        H, d, C = self.n_heads, self.head_dim, self.chunk_size
        x = out.float()

        # projections — token shift for write key
        v = self.proj_write(x).view(B, T, H, d)
        gate = self.gate_proj(x).sigmoid().view(B, T, H, 1)
        beta = self.beta_proj(x).sigmoid().view(B, T, H, 1)
        decay = -self.A_log.exp().view(1, 1, H) * F.softplus(self.alpha_proj(x) + self.dt_bias)

        rk = x.view(B, T, H, d)
        wk = F.pad(rk[:, :-1], (0, 0, 0, 0, 1, 0))  # shift along T dim

        # transpose to (B, H, T, d)
        rk, wk, v = [t.transpose(1, 2).float() for t in (rk, wk, v)]

        # scale v by beta
        v = v * beta.transpose(1, 2)

        # pad to chunk boundary
        pad = (C - (T % C)) % C
        if pad > 0:
            rk = F.pad(rk, (0, 0, 0, pad))
            wk = F.pad(wk, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            decay = F.pad(decay, (0, 0, 0, pad))

        T_pad = rk.shape[2]
        N = T_pad // C

        # reshape into chunks: (B, H, N, C, d)
        rk = rk.view(B, H, N, C, d)
        wk = wk.view(B, H, N, C, d)
        v = v.view(B, H, N, C, d)
        decay = decay.transpose(1, 2).view(B, H, N, C)

        # cumulative decay within each chunk
        decay = decay.cumsum(-1)
        decay_exp = decay.unsqueeze(-1).exp()

        # intra-chunk causal decay mask (tril before exp prevents upper triangle overflow)
        L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()

        # causal mask: strictly lower triangular (read before write)
        causal = torch.triu(torch.ones(C, C, device=x.device, dtype=torch.bool), diagonal=0)

        # intra-chunk attention
        intra = (rk @ wk.transpose(-1, -2) * L_mask).masked_fill(causal, 0)

        # chunk-by-chunk state propagation
        S = x.new_zeros(B, H, d, d)
        o = torch.zeros_like(v)

        for i in range(N):
            rk_i, wk_i, v_i = rk[:, :, i], wk[:, :, i], v[:, :, i]
            o_inter = (rk_i * decay_exp[:, :, i]) @ S
            o[:, :, i] = o_inter + intra[:, :, i] @ v_i

            # advance state
            decay_weights = (decay[:, :, i, -1, None] - decay[:, :, i]).exp().unsqueeze(-1)
            S = S * decay[:, :, i, -1, None, None].exp() + (wk_i * decay_weights).transpose(-1, -2) @ v_i
            S_norm = S.norm(dim=(-2, -1), keepdim=True)
            S = S * (S_norm.clamp(max=100.0) / S_norm.clamp(min=1e-6))

        # output gate + projection
        o = o.view(B, H, T_pad, d).transpose(1, 2)[:, :T]  # (B, T, H, d)
        o = o * gate  # per-head gating
        return out + self.out_proj(o.reshape(B, T, D)).to(out.dtype)

    def step(self, out, state=None):
        """Recurrent form with data-dependent decay.

        out: (B, D) hidden state.
        state: dict with 'W' and 'r_prev', or None.
        returns: (B, D) augmented, new state dict.
        """
        B, D = out.shape
        H, d = self.n_heads, self.head_dim

        v = self.proj_write(out).view(B, H, d)
        gate = self.gate_proj(out).sigmoid().view(B, H, 1)
        beta = self.beta_proj(out).sigmoid().view(B, H, 1)
        decay = (-self.A_log.exp() * F.softplus(self.alpha_proj(out) + self.dt_bias)).exp().view(B, H, 1, 1)

        v = v * beta
        rk = out.view(B, H, d)

        if state is not None:
            W = state["W"]
            r_prev = state["r_prev"]
        else:
            W = out.new_zeros(B, H, d, d)
            r_prev = out.new_zeros(B, H, d)

        # decay state, read, write
        W = W * decay
        read = (W * rk.unsqueeze(-1)).sum(-2)  # (B, H, d)
        W = W + v.unsqueeze(-1) * r_prev.unsqueeze(-2)

        # output gate
        read = read * gate
        read = self.out_proj(read.reshape(B, D))

        return out + read, {"W": W, "r_prev": rk}


class DeltaHebbianBlock(nn.Module):
    """Block-diagonal delta rule memory with data-dependent decay and output gate.

    Token shift: write key is previous token, read key is current.
    No Q/K projections — raw input (normalized) is the key.
    Delta rule: W_t = e^g · W_{t-1} + β(v - W·wk)·wk^T
    Output: norm(read) * silu(gate)
    """

    def __init__(self, d_model: int, num_heads: int = 8, chunk_size: int = 64):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.n_heads = num_heads
        self.head_dim = d_model // num_heads
        self.chunk_size = chunk_size

        self.proj_write = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, self.n_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # write gate
        self.beta_proj = nn.Linear(d_model, self.n_heads, bias=False)

        # data-dependent decay (Mamba-style init)
        self.alpha_proj = nn.Linear(d_model, self.n_heads, bias=False)
        dt = torch.empty(self.n_heads).uniform_(0, 1) * (torch.tensor(0.1).log() - torch.tensor(0.001).log()) + torch.tensor(0.001).log()
        dt = dt.exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True
        self.A_log = nn.Parameter(torch.empty(self.n_heads).uniform_(0, 4).log())
        self.A_log._no_weight_decay = True

        # static masks
        C = chunk_size
        self.register_buffer("causal_mask", torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0), persistent=False)
        self.register_buffer("eye_C", torch.eye(C), persistent=False)

    def forward(self, out):
        """Chunkwise parallel delta rule with data-dependent decay.

        out: (B, T, D).
        returns: (B, T, D) with memory reads added.
        """
        B, T, D = out.shape
        H, d, C = self.n_heads, self.head_dim, self.chunk_size
        x = out.float()

        # projections — token shift for write key, normalize keys
        v = self.proj_write(x).view(B, T, H, d)
        gate = self.gate_proj(x).sigmoid().view(B, T, H, 1)
        beta = self.beta_proj(x).sigmoid()
        decay = -self.A_log.exp().view(1, 1, H) * F.softplus(self.alpha_proj(x) + self.dt_bias)

        rk = F.normalize(x.view(B, T, H, d), dim=-1)
        wk = F.pad(rk[:, :-1], (0, 0, 0, 0, 1, 0))  # shift along T dim

        # transpose to (B, H, T, d)
        rk, wk, v = [t.transpose(1, 2).float() for t in (rk, wk, v)]

        # scale v and wk by beta
        v = v * beta.transpose(1, 2).unsqueeze(-1)
        wk_beta = wk * beta.transpose(1, 2).unsqueeze(-1)

        # pad to chunk boundary
        pad = (C - (T % C)) % C
        if pad > 0:
            rk = F.pad(rk, (0, 0, 0, pad))
            wk = F.pad(wk, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            wk_beta = F.pad(wk_beta, (0, 0, 0, pad))
            decay = F.pad(decay, (0, 0, 0, pad))

        T_pad = rk.shape[2]
        N = T_pad // C

        # reshape into chunks: (B, H, N, C, d)
        rk = rk.view(B, H, N, C, d)
        wk = wk.view(B, H, N, C, d)
        v = v.view(B, H, N, C, d)
        wk_beta = wk_beta.view(B, H, N, C, d)
        decay = decay.transpose(1, 2).view(B, H, N, C)

        # cumulative decay within each chunk
        decay = decay.cumsum(-1)
        decay_exp = decay.unsqueeze(-1).exp()

        # intra-chunk causal decay mask
        L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()

        # WY correction
        causal_upper = torch.triu(torch.ones(C, C, device=x.device, dtype=torch.bool), diagonal=0)
        A = -(wk_beta @ wk.transpose(-1, -2) * L_mask).masked_fill(causal_upper, 0)
        A = A.clone()
        for i in range(1, C):
            A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :i].clone().unsqueeze(-1) * A[..., :i, :i].clone()).sum(-2)
        A = A + torch.eye(C, device=x.device)

        v = A @ v
        wk_cumdecay = A @ (wk_beta * decay_exp)

        # chunk-by-chunk state propagation
        S = x.new_zeros(B, H, d, d)
        o = torch.zeros_like(v)
        causal_mask = torch.triu(torch.ones(C, C, device=x.device, dtype=torch.bool), diagonal=1)

        for i in range(N):
            rk_i, wk_i, v_i = rk[:, :, i], wk[:, :, i], v[:, :, i]
            attn = (rk_i @ wk_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill(causal_mask, 0)
            v_new = v_i - wk_cumdecay[:, :, i] @ S
            o[:, :, i] = (rk_i * decay[:, :, i, :, None].exp()) @ S + attn @ v_new
            decay_weights = (decay[:, :, i, -1, None] - decay[:, :, i]).exp().unsqueeze(-1)
            S = S * decay[:, :, i, -1, None, None].exp() + (wk_i * decay_weights).transpose(-1, -2) @ v_new
            S_norm = S.norm(dim=(-2, -1), keepdim=True)
            S = S * (S_norm.clamp(max=100.0) / S_norm.clamp(min=1e-6))

        # output gate + projection
        o = o.view(B, H, T_pad, d).transpose(1, 2)[:, :T]  # (B, T, H, d)
        o = o * gate  # per-head gating
        return out + self.out_proj(o.reshape(B, T, D)).to(out.dtype)

    def step(self, out, state=None):
        """Sequential recurrence with data-dependent decay and output gate.

        out: (B, D).
        returns: (B, D), new state.
        """
        B, D = out.shape
        H, d = self.n_heads, self.head_dim

        v = self.proj_write(out).view(B, H, d)
        gate = self.gate_proj(out).sigmoid().view(B, H, 1)
        beta = self.beta_proj(out).sigmoid().unsqueeze(-1)  # (B, H, 1)
        decay = (-self.A_log.exp() * F.softplus(self.alpha_proj(out) + self.dt_bias)).exp().view(B, H, 1, 1)

        rk = F.normalize(out.view(B, H, d), dim=-1)

        if state is not None:
            W = state["W"]
            wk = state["wk"]
        else:
            W = out.new_zeros(B, H, d, d)
            wk = out.new_zeros(B, H, d)

        # decay state
        W = W * decay

        # delta rule write with previous key, scaled by beta
        v_err = (v - (W * wk.unsqueeze(-1)).sum(-2)) * beta
        W = W + wk.unsqueeze(-1) * v_err.unsqueeze(-2)

        # read with current key
        read = (W * rk.unsqueeze(-1)).sum(-2)  # (B, H, d)

        # output gate
        read = read * gate  # per-head gating
        read = self.out_proj(read.reshape(B, D))

        return out + read, {"W": W, "wk": rk}
