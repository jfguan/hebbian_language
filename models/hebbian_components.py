"""Reusable components for Hebbian models.

- CausalConv: causal depthwise conv1d for local token mixing
- GatedMLP: SwiGLU gated projections for channel mixing
- HebbianBlock: outer-product associative memory for long-range binding
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


class HebbianBlock(nn.Module):
    """Block-diagonal associative memory with delta rule.

    n_heads independent head_dim×head_dim memory matrices.
    Input is the key. proj_write produces values. proj_read is the output projection.
    Delta rule: W_t = γW_{t-1} + β_t(v_t - W_{t-1}k_t)k_t^T
    """

    def __init__(self, d_model: int, head_dim: int = 128):
        super().__init__()
        assert d_model % head_dim == 0
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim

        self.proj_write = nn.Linear(d_model, d_model, bias=False)
        self.proj_read = nn.Linear(d_model, d_model, bias=False)

        # per-head decay: σ(4.6) ≈ 0.99
        self.decay = nn.Parameter(torch.full((self.n_heads,), 4.6))

        # data-dependent learning rate for delta rule
        self.proj_beta = nn.Linear(d_model, self.n_heads, bias=False)

    def forward(self, out):
        """Chunkwise parallel delta rule.

        out: (B, T, D) hidden states (used as keys).

        returns: (B, T, D) augmented with memory reads.
        """
        B, T, D = out.shape
        H, d = self.n_heads, self.head_dim
        C = 64
        out32 = out.float()

        # projections
        k = F.normalize(out32.view(B, T, H, d), dim=-1).transpose(1, 2).float()  # (B, H, T, d)
        v = self.proj_write(out32).view(B, T, H, d).transpose(1, 2).float()
        beta = torch.sigmoid(self.proj_beta(out32)).transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)

        # scale v and k by beta
        v = v * beta
        k_beta = k * beta

        # pad to chunk boundary
        pad_len = (C - (T % C)) % C
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            k_beta = F.pad(k_beta, (0, 0, 0, pad_len))
        T_padded = k.shape[2]
        num_chunks = T_padded // C

        # precompute decay masks — constant per head, reused across all chunks
        log_gamma = torch.sigmoid(self.decay).log()  # (H,)
        positions = torch.arange(C, device=out.device)
        cum_decay = (positions + 1) * log_gamma.view(H, 1)  # (H, C)
        L_mask = (cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)).tril().exp().tril()  # (H, C, C)
        L_mask = L_mask.view(1, H, 1, C, C)  # broadcastable with (B, H, num_chunks, C, C)
        decay_exp = cum_decay.unsqueeze(-1).exp().view(1, H, 1, C, 1)  # broadcastable
        chunk_total_decay = cum_decay[:, -1].exp().view(1, H, 1, 1)  # decay across full chunk
        causal_mask = torch.triu(torch.ones(C, C, device=out.device, dtype=torch.bool), diagonal=1)

        # reshape into chunks: (B, H, num_chunks, C, d)
        k = k.view(B, H, num_chunks, C, d)
        v = v.view(B, H, num_chunks, C, d)
        k_beta = k_beta.view(B, H, num_chunks, C, d)

        # WY pre-processing: solve (I + A) for corrected values
        diag_mask = torch.triu(torch.ones(C, C, device=out.device, dtype=torch.bool))
        A = -(k_beta @ k.transpose(-1, -2) * L_mask).masked_fill(diag_mask, 0)
        A = A.clone()
        for i in range(1, C):
            A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :i].clone().unsqueeze(-1) * A[..., :i, :i].clone()).sum(-2)
        A = A + torch.eye(C, device=out.device)

        v = A @ v
        k_cumdecay = A @ (k_beta * decay_exp)

        # chunk-by-chunk state propagation
        S = out32.new_zeros(B, H, d, d)
        o = torch.zeros_like(v)
        L = L_mask.squeeze(2)           # (1, H, C, C)
        de = decay_exp.squeeze(2)       # (1, H, C, 1)
        dw = ((cum_decay[:, -1:] - cum_decay).exp()).unsqueeze(-1)  # (H, C, 1)

        for i in range(num_chunks):
            k_i, v_i = k[:, :, i], v[:, :, i]
            attn = (k_i @ k_i.transpose(-1, -2) * L).masked_fill(causal_mask, 0)
            v_new = v_i - k_cumdecay[:, :, i] @ S
            o[:, :, i] = (k_i * de) @ S + attn @ v_new
            S = chunk_total_decay * S + (k_i * dw).transpose(-1, -2) @ v_new

        # flatten chunks, unpad
        o = o.view(B, H, T_padded, d).transpose(1, 2).reshape(B, T_padded, D)[:, :T]
        return out + self.proj_read(o).to(out.dtype)

    def step(self, out, state=None):
        """Recurrent form for inference.

        out: (B, D) hidden state (used as key).
        state: dict with 'W', or None.

        returns: (B, D) augmented, new state dict.
        """
        B, D = out.shape
        H, d = self.n_heads, self.head_dim

        k = F.normalize(out.view(B * H, d), dim=-1)
        v = self.proj_write(out).view(B * H, d)
        beta = torch.sigmoid(self.proj_beta(out)).view(B * H, 1, 1)
        gamma = torch.sigmoid(self.decay).repeat(B).view(B * H, 1, 1)
        W = state["W"].view(B * H, d, d) if state is not None else out.new_zeros(B * H, d, d)

        # read with key
        read = (W @ k.unsqueeze(-1)).squeeze(-1)

        # delta rule: W = γW + β(v - Wk)k^T
        error = (v - read).unsqueeze(-1)
        W.mul_(gamma).add_(beta * error @ k.unsqueeze(-2))

        read = read.view(B, D)
        return out + self.proj_read(read), {"W": W.view(B, H, d, d)}
