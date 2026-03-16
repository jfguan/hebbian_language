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
    """D*D associative memory: W_t = γW_{t-1} + v_t⊗k_{t-1}, read_t = W_t·q_t.

    Chunkwise parallel O(TC·D + T·D²) for training, recurrent O(D²) for inference.
    """

    def __init__(self, d_model: int, chunk_size: int = 64, memory_alpha: float = 0.03):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size

        self.proj_write = nn.Linear(d_model, d_model, bias=False)
        self.proj_read = nn.Linear(d_model, d_model, bias=False)
        self.decay = nn.Parameter(torch.tensor(4.6))  # σ(4.6) ≈ 0.99
        self.log_alpha = nn.Parameter(torch.tensor(memory_alpha).log())

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(self, out):
        """Chunkwise parallel form.

        out: (B, T, D) hidden states.

        returns: (B, T, D) augmented with memory reads.
        """
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

            # Inter-chunk: γ^l * (W_prev @ rk_c[l])
            inter = torch.matmul(W, rk_c.transpose(1, 2)).transpose(1, 2)
            inter = inter * (gamma ** p)[None, :, None]

            # Intra-chunk: (M ⊙ S) @ v
            S = torch.bmm(rk_c, wk_c.transpose(1, 2))
            diffs = (p[:, None] - 1 - p[None, :]).clamp(min=0)
            causal = p[:, None] > p[None, :]
            M = torch.exp(diffs * log_gamma) * causal
            intra = torch.bmm(S * M, v_c)

            reads_list.append(inter + intra)

            # Advance W: γ^Ci · W + Σ_l γ^(Ci-1-l) · v[l] ⊗ wk[l]
            gw = (gamma ** (Ci - 1 - p))[None, :, None]
            W = gamma ** Ci * W + torch.bmm((v_c * gw).transpose(1, 2), wk_c)

        reads = torch.cat(reads_list, dim=1)
        return out + self.alpha * self.proj_read(reads).to(out.dtype)

    def step(self, out, state=None):
        """Recurrent form.

        out: (B, D) hidden state.
        state: dict with 'W' and 'r_prev', or None.

        returns: (B, D) augmented, new state dict.
        """
        B, D = out.shape

        if state is None:
            W = out.new_zeros(B, D, D)
            r_prev = out.new_zeros(B, D)
        else:
            W = state["W"]
            r_prev = state["r_prev"]

        gamma = torch.sigmoid(self.decay)
        write = torch.einsum("bi,bj->bij", self.proj_write(out), r_prev)
        read = torch.einsum("bij,bj->bi", W, out)
        W = gamma * W + write

        augmented = out + self.alpha * self.proj_read(read)

        new_state = {"W": W, "r_prev": out}
        return augmented, new_state


class DeltaHebbianBlock(nn.Module):
    """Block-diagonal associative memory with delta rule.

    n_heads independent head_dim×head_dim memory matrices.
    Write key is previous timestep, read key is current.
    Delta rule: W_t = γW_{t-1} + β_t(v_t - W_{t-1}·wk_t)·wk_t^T
    """

    def __init__(self, d_model: int, head_dim: int = 128, chunk_size: int = 64):
        super().__init__()
        assert d_model % head_dim == 0
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        self.chunk_size = chunk_size

        self.proj_write = nn.Linear(d_model, d_model, bias=False)
        self.proj_read = nn.Linear(d_model, d_model, bias=False)

        # per-head decay: σ(4.6) ≈ 0.99
        self.decay = nn.Parameter(torch.full((self.n_heads,), 4.6))

        # data-dependent write gate
        self.proj_beta = nn.Linear(d_model, self.n_heads, bias=False)

        # static masks
        C = chunk_size
        self.register_buffer("positions", torch.arange(C), persistent=False)
        self.register_buffer("causal_mask", torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0), persistent=False)
        self.register_buffer("diag_mask", torch.triu(torch.ones(C, C, dtype=torch.bool)), persistent=False)

    def forward(self, out):
        """Chunkwise parallel delta rule.

        out: (B, T, D) hidden states (used as keys).

        returns: (B, T, D) augmented with memory reads.
        """
        B, T, D = out.shape
        H, d = self.n_heads, self.head_dim
        C = self.chunk_size
        out32 = out.float()

        # projections — write key is previous position, read key is current
        rk = F.normalize(out32.view(B, T, H, d), dim=-1).transpose(1, 2).float()
        wk = F.pad(rk[:, :, :-1], (0, 0, 1, 0))
        v = self.proj_write(out32).view(B, T, H, d).transpose(1, 2).float()
        beta = torch.sigmoid(self.proj_beta(out32)).transpose(1, 2).unsqueeze(-1)

        # scale v and wk by beta
        v = v * beta
        wk_beta = wk * beta

        # pad to chunk boundary
        pad_len = (C - (T % C)) % C
        if pad_len > 0:
            rk = F.pad(rk, (0, 0, 0, pad_len))
            wk = F.pad(wk, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            wk_beta = F.pad(wk_beta, (0, 0, 0, pad_len))
        T_padded = rk.shape[2]
        num_chunks = T_padded // C

        # decay masks — depend on learned decay but not on input
        log_gamma = torch.sigmoid(self.decay).log()
        cum_decay = (self.positions + 1) * log_gamma.view(H, 1)
        L_mask = (cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)).tril().exp().tril()
        L_mask = L_mask.view(1, H, 1, C, C)
        decay_exp = cum_decay.unsqueeze(-1).exp().view(1, H, 1, C, 1)
        chunk_total_decay = cum_decay[:, -1].exp().view(1, H, 1, 1)

        # reshape into chunks: (B, H, num_chunks, C, d)
        rk = rk.view(B, H, num_chunks, C, d)
        wk = wk.view(B, H, num_chunks, C, d)
        v = v.view(B, H, num_chunks, C, d)
        wk_beta = wk_beta.view(B, H, num_chunks, C, d)

        # WY pre-processing: solve (I + A) for corrected values
        A = -(wk_beta @ wk.transpose(-1, -2) * L_mask).masked_fill(self.diag_mask, 0)
        A = A.clone()
        for i in range(1, C):
            A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :i].clone().unsqueeze(-1) * A[..., :i, :i].clone()).sum(-2)
        A = A + torch.eye(C, device=out.device)

        v = A @ v
        wk_cumdecay = A @ (wk_beta * decay_exp)

        # chunk-by-chunk state propagation
        S = out32.new_zeros(B, H, d, d)
        o = torch.zeros_like(v)
        L = L_mask.squeeze(2)
        de = decay_exp.squeeze(2)
        dw = ((cum_decay[:, -1:] - cum_decay).exp()).unsqueeze(-1)

        for i in range(num_chunks):
            rk_i, wk_i, v_i = rk[:, :, i], wk[:, :, i], v[:, :, i]
            attn = (rk_i @ wk_i.transpose(-1, -2) * L).masked_fill(self.causal_mask, 0)
            v_new = v_i - wk_cumdecay[:, :, i] @ S
            o[:, :, i] = (rk_i * de) @ S + attn @ v_new
            S = chunk_total_decay * S + (wk_i * dw).transpose(-1, -2) @ v_new

        # flatten chunks, unpad
        o = o.view(B, H, T_padded, d).transpose(1, 2).reshape(B, T_padded, D)[:, :T]
        return out + self.proj_read(o).to(out.dtype)

    def step(self, out, state=None):
        """Recurrent form for inference.

        out: (B, D) hidden state (used as key).
        state: dict with 'W' and 'k_prev', or None.

        returns: (B, D) augmented, new state dict.
        """
        B, D = out.shape
        H, d = self.n_heads, self.head_dim

        rk = F.normalize(out.view(B * H, d), dim=-1)
        v = self.proj_write(out).view(B * H, d)
        beta = torch.sigmoid(self.proj_beta(out)).view(B * H, 1, 1)
        gamma = torch.sigmoid(self.decay).repeat(B).view(B * H, 1, 1)

        if state is not None:
            W = state["W"].view(B * H, d, d)
            wk = state["k_prev"].view(B * H, d)
        else:
            W = out.new_zeros(B * H, d, d)
            wk = out.new_zeros(B * H, d)

        # read with current key
        read = (W @ rk.unsqueeze(-1)).squeeze(-1)

        # delta rule write with previous key: W = γW + β(v - W@wk) ⊗ wk
        error = (v - (W @ wk.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)
        W = gamma * W + beta * error @ wk.unsqueeze(-2)

        read = read.view(B, D)
        return out + self.proj_read(read), {"W": W.view(B, H, d, d), "k_prev": rk.view(B, H, d)}
