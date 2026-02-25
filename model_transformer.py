"""GPT-2 style transformer using HuggingFace GPT2LMHeadModel.

Configured to ~18M params for codeparrot baseline.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config


def make_config(vocab_size=384, max_seq_len=2048):
    """~18M param GPT-2 config."""
    return GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_seq_len,
        n_embd=512,
        n_layer=6,
        n_head=8,
        n_inner=2048,  # 4x expansion
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=True,
    )


class Transformer(torch.nn.Module):
    def __init__(self, cfg=None, vocab_size=384, max_seq_len=2048):
        super().__init__()
        if cfg is None:
            cfg = make_config(vocab_size, max_seq_len)
        self.model = GPT2LMHeadModel(cfg)

    def forward(self, input_ids, targets=None):
        out = self.model(input_ids)
        logits = out.logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
