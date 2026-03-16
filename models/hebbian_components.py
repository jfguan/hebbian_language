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


class HebbianBlock(nn.Module):
    """Block-diagonal associative memory: W_t = γW_{t-1} + v_t⊗k_{t-1}, read_t = W_t·q_t.

    When head_dim is None (default), uses a single D×D matrix.
    When head_dim is set, uses n_heads independent head_dim×head_dim matrices.
    Chunkwise parallel for training, recurrent for inference.
    """

    def __init__(self, d_model: int, chunk_size: int = 64, memory_alpha: float = 0.03, head_dim: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size

        if head_dim is not None:
            assert d_model % head_dim == 0
            self.head_dim = head_dim
            self.n_heads = d_model // head_dim
        else:
            self.head_dim = d_model
            self.n_heads = 1

        self.proj_write = nn.Linear(d_model, d_model, bias=False)
        self.proj_read = nn.Linear(d_model, d_model, bias=False)
        self.decay = nn.Parameter(torch.full((self.n_heads,), 4.6))  # σ(4.6) ≈ 0.99
        self.log_alpha = nn.Parameter(torch.full((self.n_heads,), memory_alpha).log())

    def forward(self, out):
        """Chunkwise parallel form.

        out: (B, T, D) hidden states.
        returns: (B, T, D) augmented with memory reads.
        """
        B, T, D = out.shape
        H, d, C = self.n_heads, self.head_dim, self.chunk_size
        out32 = out.float()

        gamma = torch.sigmoid(self.decay)                                  # (H,)
        log_gamma = gamma.log()                                                # (H,)

        v = self.proj_write(out32).view(B, T, H, d).transpose(1, 2)     # (B, H, T, d)
        rk = out32.view(B, T, H, d).transpose(1, 2)                      # (B, H, T, d)
        wk = F.pad(rk[:, :, :-1], (0, 0, 1, 0))                         # (B, H, T, d)

        # -- flatten B*H for batched matmuls --
        BH = B * H
        gamma_bh = gamma.repeat(B)                                         # (BH,)
        log_gamma_bh = log_gamma.repeat(B)                                 # (BH,)
        v = v.reshape(BH, T, d)
        rk = rk.reshape(BH, T, d)
        wk = wk.reshape(BH, T, d)

        W = out32.new_zeros(BH, d, d)
        reads_list = []

        for start in range(0, T, C):
            end = min(start + C, T)
            Ci = end - start
            p = torch.arange(Ci, device=out.device)

            rk_c, wk_c, v_c = rk[:, start:end], wk[:, start:end], v[:, start:end]

            # Inter-chunk: γ^l * (W_prev @ rk_c[l])
            inter = torch.matmul(W, rk_c.transpose(1, 2)).transpose(1, 2)
            inter = inter * (gamma_bh[:, None, None] ** p[None, :, None])

            # Intra-chunk: (M ⊙ S) @ v
            S = torch.bmm(rk_c, wk_c.transpose(1, 2))
            diffs = (p[:, None] - 1 - p[None, :]).clamp(min=0)
            causal = p[:, None] > p[None, :]
            M = torch.exp(diffs[None] * log_gamma_bh[:, None, None]) * causal[None]
            intra = torch.bmm(S * M, v_c)

            reads_list.append(inter + intra)

            # Advance W: γ^Ci · W + Σ_l γ^(Ci-1-l) · v[l] ⊗ wk[l]
            gw = gamma_bh[:, None, None] ** (Ci - 1 - p)[None, :, None]
            W = gamma_bh[:, None, None] ** Ci * W + torch.bmm((v_c * gw).transpose(1, 2), wk_c)

        reads = torch.cat(reads_list, dim=1).view(B, H, T, d)
        alpha = self.log_alpha.exp().view(1, H, 1, 1)
        reads = (alpha * reads).transpose(1, 2).reshape(B, T, D)
        return out + self.proj_read(reads).to(out.dtype)

    def step(self, out, state=None):
        """Recurrent form.

        out: (B, D) hidden state.
        state: dict with 'W' and 'r_prev', or None.
        returns: (B, D) augmented, new state dict.
        """
        B, D = out.shape
        H, d = self.n_heads, self.head_dim
        BH = B * H

        if state is None:
            W = out.new_zeros(BH, d, d)
            r_prev = out.new_zeros(BH, d)
        else:
            W = state["W"]
            r_prev = state["r_prev"]

        gamma = torch.sigmoid(self.decay).repeat(B).view(BH, 1, 1)     # (BH, 1, 1)
        v = self.proj_write(out).view(BH, d)
        rk = out.view(BH, d)

        write = torch.einsum("bi,bj->bij", v, r_prev)
        read = torch.einsum("bij,bj->bi", W, rk)
        W = gamma * W + write

        alpha = self.log_alpha.exp().repeat(B).view(BH, 1)              # (BH, 1)
        read = (alpha * read).view(B, D)
        augmented = out + self.proj_read(read)

        new_state = {"W": W, "r_prev": rk}
        return augmented, new_state


class DeltaHebbianBlock(nn.Module):
    """Block-diagonal delta rule memory with scalar decay.

    Same as HebbianBlock but with error-corrective writes:
    W_t = γ · W_{t-1} + (v_t - W_{t-1} · wk_t) ⊗ wk_t

    The delta rule only writes what the memory doesn't already know.
    Chunkwise parallel for training, recurrent for inference.
    """

    def __init__(self, d_model: int, head_dim: int = 256, chunk_size: int = 64, memory_alpha: float = 0.03):
        super().__init__()
        assert d_model % head_dim == 0
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        self.chunk_size = chunk_size

        self.proj_write = nn.Linear(d_model, d_model, bias=False)
        self.proj_read = nn.Linear(d_model, d_model, bias=False)
        self.decay = nn.Parameter(torch.full((self.n_heads,), 4.6))  # σ(4.6) ≈ 0.99
        self.log_alpha = nn.Parameter(torch.full((self.n_heads,), memory_alpha).log())

        # static masks
        C = chunk_size
        self.register_buffer("causal_mask", torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0), persistent=False)
        self.register_buffer("eye_C", torch.eye(C), persistent=False)

    def forward(self, out):
        """Chunkwise parallel delta rule.

        out: (B, T, D).
        returns: (B, T, D) with memory reads added.
        """
        B, T, D = out.shape
        H, d, C = self.n_heads, self.head_dim, self.chunk_size
        x = out.float()

        gamma = torch.sigmoid(self.decay)         # (H,)
        log_gamma = gamma.log()                    # (H,)

        # -- projections: normalize keys, then shift for write key --
        v = self.proj_write(x).view(B, T, H, d).transpose(1, 2)       # (B, H, T, d)
        rk = F.normalize(x.view(B, T, H, d), dim=-1).transpose(1, 2)  # (B, H, T, d)
        wk = F.pad(rk[:, :, :-1], (0, 0, 1, 0))                      # (B, H, T, d)

        # -- flatten B*H for batched matmuls --
        BH = B * H
        gamma_bh = gamma.repeat(B)                                     # (BH,)
        log_gamma_bh = log_gamma.repeat(B)                             # (BH,)
        v = v.reshape(BH, T, d)
        rk = rk.reshape(BH, T, d)
        wk = wk.reshape(BH, T, d)

        # -- pad to chunk boundary --
        pad = (C - (T % C)) % C
        if pad > 0:
            rk = F.pad(rk, (0, 0, 0, pad))
            wk = F.pad(wk, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))

        T_pad = rk.shape[1]
        N = T_pad // C

        # -- reshape into chunks: (BH, N, C, d) --
        rk = rk.view(BH, N, C, d)
        wk = wk.view(BH, N, C, d)
        v = v.view(BH, N, C, d)

        # -- decay mask: γ^(i-j) lower triangular, same for all chunks --
        p = torch.arange(C, device=out.device)
        diffs = (p[:, None] - p[None, :]).clamp(min=0).float()
        L = torch.exp(diffs * log_gamma_bh[:, None, None]).tril()      # (BH, C, C)

        # -- WY correction: solve (I + A)x = b --
        A = -(wk @ wk.transpose(-1, -2) * L[:, None]).masked_fill(self.causal_mask, 0)
        A = A.clone()
        for i in range(1, C):
            A[..., i, :i] = A[..., i, :i].clone() + (
                A[..., i, :i].clone().unsqueeze(-1) * A[..., :i, :i].clone()
            ).sum(-2)
        A = A + self.eye_C

        # -- precompute per-position decay weights --
        gp = gamma_bh[:, None, None] ** p[None, :, None]              # (BH, C, 1): γ^0, γ^1, ..., γ^(C-1)
        gp_tail = gamma_bh[:, None, None] ** (C - 1 - p)[None, :, None]  # (BH, C, 1): γ^(C-1), ..., γ^0
        gamma_C = gamma_bh[:, None, None] ** C                        # (BH, 1, 1)

        intra = (rk @ wk.transpose(-1, -2) * L[:, None]).masked_fill(self.causal_mask, 0)
        rk_g = rk * gp[:, None]                                       # (BH, N, C, d)
        wk_g = (wk * gp_tail[:, None]).transpose(-1, -2)              # (BH, N, d, C)

        # -- chunk-by-chunk state propagation --
        S = x.new_zeros(BH, d, d)
        o = x.new_zeros(BH, N, C, d)

        for i in range(N):
            v_corr = A[:, i] @ v[:, i]
            wk_corr = A[:, i] @ (wk[:, i] * gp[:, None, 0])
            v_new = v_corr - wk_corr @ S
            o[:, i] = rk_g[:, i] @ S + intra[:, i] @ v_new
            S = gamma_C * S + wk_g[:, i] @ v_new

        # -- output --
        alpha = self.log_alpha.exp().view(1, H, 1, 1)
        o = o.view(B, H, T_pad, d)
        o = (alpha * o).transpose(1, 2).reshape(B, T_pad, D)[:, :T]
        return out + self.proj_read(o).to(out.dtype)

    def step(self, out, state=None):
        """Sequential recurrence: W = γW + (v - W·wk)⊗wk, read = W·rk.

        out: (B, D).
        returns: (B, D), new state.
        """
        B, D = out.shape
        H, d = self.n_heads, self.head_dim
        BH = B * H

        v = self.proj_write(out).view(BH, d)
        rk = F.normalize(out.view(BH, d), dim=-1)
        gamma = torch.sigmoid(self.decay).repeat(B).view(BH, 1, 1)

        if state is not None:
            W = state["W"].view(BH, d, d)
            wk = state["wk"].view(BH, d)
        else:
            W = out.new_zeros(BH, d, d)
            wk = out.new_zeros(BH, d)

        W = gamma * W
        read = (W @ rk.unsqueeze(-1)).squeeze(-1)
        error = v - (W @ wk.unsqueeze(-1)).squeeze(-1)
        W = W + error.unsqueeze(-1) @ wk.unsqueeze(-2)

        alpha = self.log_alpha.exp().repeat(B).view(BH, 1)
        read = (alpha * read).view(B, D)
        return out + self.proj_read(read), {"W": W.view(B, H, d, d), "wk": rk.view(B, H, d)}
