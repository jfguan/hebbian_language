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
