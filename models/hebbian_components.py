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
    """Block-diagonal delta rule memory with input-dependent decay.

    Maintains n_heads independent head_dim × head_dim memory matrices.
    Each token decides how much to forget (decay) and how strongly to write (beta).
    The delta rule only writes what the memory doesn't already know.

    Recurrence (per head):
        decay_t = exp(-A · softplus(proj_alpha(x_t) + dt_bias))
        W_t = decay_t · W_{t-1} + β_t · (v_t - W_{t-1} · wk_t) · wk_t^T
        read_t = W_t · rk_t

    where rk = current key (read), wk = previous key (write), v = proj_write(x).
    Training uses chunkwise parallel form; inference uses sequential recurrence.
    """

    def __init__(self, d_model: int, head_dim: int = 128, chunk_size: int = 64):
        super().__init__()
        assert d_model % head_dim == 0
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        self.chunk_size = chunk_size

        # value and output projections
        self.proj_write = nn.Linear(d_model, d_model, bias=False)
        self.proj_read = nn.Linear(d_model, d_model, bias=False)

        # input-dependent decay: g = -A · softplus(proj_alpha(x) + dt_bias)
        self.A_log = nn.Parameter(torch.empty(self.n_heads).uniform_(0, 16).log())
        self.proj_alpha = nn.Linear(d_model, self.n_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.n_heads))

        # input-dependent write gate
        self.proj_beta = nn.Linear(d_model, self.n_heads, bias=False)

        # static masks (registered as buffers so they move with .to(device))
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

        # -- projections --
        rk = F.normalize(x.view(B, T, H, d), dim=-1).transpose(1, 2)  # read key: current position
        wk = F.pad(rk[:, :, :-1], (0, 0, 1, 0))                      # write key: previous position
        v = self.proj_write(x).view(B, T, H, d).transpose(1, 2)       # value to store
        beta = torch.sigmoid(self.proj_beta(x)).transpose(1, 2).unsqueeze(-1)  # write gate (B, H, T, 1)

        # -- input-dependent decay --
        log_decay = -self.A_log.exp().view(1, 1, H) * F.softplus(self.proj_alpha(x) + self.dt_bias)
        log_decay = log_decay.transpose(1, 2)  # (B, H, T)

        # -- scale by beta --
        v_scaled = v * beta
        wk_scaled = wk * beta

        # -- pad to chunk boundary --
        pad = (C - (T % C)) % C
        if pad > 0:
            rk = F.pad(rk, (0, 0, 0, pad))
            wk = F.pad(wk, (0, 0, 0, pad))
            v_scaled = F.pad(v_scaled, (0, 0, 0, pad))
            wk_scaled = F.pad(wk_scaled, (0, 0, 0, pad))
            log_decay = F.pad(log_decay, (0, pad))

        T_pad = rk.shape[2]
        N = T_pad // C  # number of chunks

        # -- reshape into chunks: (B, H, N, C, d) --
        rk = rk.view(B, H, N, C, d)
        wk = wk.view(B, H, N, C, d)
        v_scaled = v_scaled.view(B, H, N, C, d)
        wk_scaled = wk_scaled.view(B, H, N, C, d)
        log_decay = log_decay.view(B, H, N, C)

        # -- decay masks (per chunk, input-dependent) --
        cum_decay = log_decay.cumsum(-1)                                           # (B, H, N, C)
        L_mask = (cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)).tril().exp().tril()  # (B, H, N, C, C)
        decay_exp = cum_decay.unsqueeze(-1).exp()                                  # (B, H, N, C, 1)
        chunk_end_decay = cum_decay[:, :, :, -1].exp()                             # (B, H, N)

        # -- WY correction: solve (I + A)x = b via forward substitution --
        A = -(wk_scaled @ wk.transpose(-1, -2) * L_mask).masked_fill(self.causal_mask, 0)
        A = A.clone()
        for i in range(1, C):
            A[..., i, :i] = A[..., i, :i].clone() + (
                A[..., i, :i].clone().unsqueeze(-1) * A[..., :i, :i].clone()
            ).sum(-2)
        A = A + self.eye_C

        # -- precompute loop-invariant terms --
        intra_attn = (rk @ wk.transpose(-1, -2) * L_mask).masked_fill(self.causal_mask, 0)  # (B, H, N, C, C)
        rk_decay = rk * decay_exp                                                            # (B, H, N, C, d)
        tail_decay = (cum_decay[:, :, :, -1:] - cum_decay).unsqueeze(-1).exp()               # (B, H, N, C, 1)
        wk_pos = (wk * tail_decay).transpose(-1, -2)                                         # (B, H, N, d, C)

        # -- chunk-by-chunk state propagation (only S is sequential) --
        S = x.new_zeros(B, H, d, d)
        o = x.new_zeros(B, H, N, C, d)

        for i in range(N):
            v_corr = A[:, :, i] @ v_scaled[:, :, i]
            wk_corr = A[:, :, i] @ (wk_scaled[:, :, i] * decay_exp[:, :, i])
            v_new = v_corr - wk_corr @ S
            o[:, :, i] = rk_decay[:, :, i] @ S + intra_attn[:, :, i] @ v_new
            S = chunk_end_decay[:, :, i, None, None] * S + wk_pos[:, :, i] @ v_new

        # -- output: flatten chunks, unpad, project --
        o = o.view(B, H, T_pad, d).transpose(1, 2).reshape(B, T_pad, D)[:, :T]
        return out + self.proj_read(o).to(out.dtype)

    def step(self, out, state=None):
        """Sequential recurrence for inference.

        out: (B, D).
        returns: (B, D), new state.
        """
        B, D = out.shape
        H, d = self.n_heads, self.head_dim

        # projections
        rk = F.normalize(out.view(B * H, d), dim=-1)
        v = self.proj_write(out).view(B * H, d)
        beta = torch.sigmoid(self.proj_beta(out)).view(B * H, 1, 1)
        log_decay = -self.A_log.exp() * F.softplus(self.proj_alpha(out) + self.dt_bias)
        decay = log_decay.exp().view(B * H, 1, 1)

        # load or init state
        if state is not None:
            W = state["W"].view(B * H, d, d)
            wk = state["k_prev"].view(B * H, d)
        else:
            W = out.new_zeros(B * H, d, d)
            wk = out.new_zeros(B * H, d)

        # decay, read, then write
        W = decay * W
        read = (W @ rk.unsqueeze(-1)).squeeze(-1)
        error = (v - (W @ wk.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)
        W = W + beta * error @ wk.unsqueeze(-2)

        read = read.view(B, D)
        return out + self.proj_read(read), {"W": W.view(B, H, d, d), "k_prev": rk.view(B, H, d)}
