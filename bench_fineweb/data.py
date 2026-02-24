"""FineWeb data loading for Conv+Heb benchmark.

Downloads pre-tokenized FineWeb10B chunks (GPT-2 tokenizer, uint16)
from HuggingFace, matching the modded-nanogpt format.

Usage:
    uv run bench_fineweb/data.py            # download 10 chunks (~1B tokens)
    uv run bench_fineweb/data.py --chunks 5 # download 5 chunks
"""

import os
import sys
import struct
import numpy as np


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "fineweb10B")
HEADER_SIZE = 256  # int32 header


def download(num_chunks=10):
    """Download pre-tokenized FineWeb chunks from HuggingFace."""
    from huggingface_hub import hf_hub_download

    os.makedirs(DATA_DIR, exist_ok=True)

    def get(fname):
        if not os.path.exists(os.path.join(DATA_DIR, fname)):
            print(f"Downloading {fname}...")
            hf_hub_download(
                repo_id="kjj0/fineweb10B-gpt2",
                filename=fname,
                repo_type="dataset",
                local_dir=DATA_DIR,
            )

    # Validation chunk
    get("fineweb_val_%06d.bin" % 0)
    # Training chunks
    for i in range(1, num_chunks + 1):
        get("fineweb_train_%06d.bin" % i)
    print(f"Downloaded val + {num_chunks} train chunks to {DATA_DIR}")


def load_tokens(path):
    """Load a binary token file (256 int32 header + uint16 tokens)."""
    with open(path, "rb") as f:
        header = struct.unpack("256i", f.read(HEADER_SIZE * 4))
        assert header[0] == 20240520, f"Bad magic: {header[0]}"
        n_tokens = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
        assert len(tokens) == n_tokens
    return tokens


def load_dataset():
    """Load all available train/val token files."""
    import glob

    val_files = sorted(glob.glob(os.path.join(DATA_DIR, "fineweb_val_*.bin")))
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, "fineweb_train_*.bin")))

    if not val_files or not train_files:
        raise FileNotFoundError(
            f"No data found in {DATA_DIR}. Run: uv run bench_fineweb/data.py"
        )

    print(f"Loading {len(train_files)} train chunks + {len(val_files)} val chunks...")
    val_tokens = np.concatenate([load_tokens(f) for f in val_files])
    train_tokens = np.concatenate([load_tokens(f) for f in train_files])
    print(f"  Train: {len(train_tokens):,} tokens | Val: {len(val_tokens):,} tokens")

    return {
        "train": train_tokens,
        "val": val_tokens,
        "vocab_size": 50304,  # GPT-2 50257 padded for GPU efficiency
    }


class DataLoader:
    """Simple random-batch data loader from a flat token array."""

    def __init__(self, tokens, batch_size, seq_len):
        import torch
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.batch_size = batch_size
        self.seq_len = seq_len

    def batch(self):
        import torch
        idx = torch.randint(len(self.tokens) - self.seq_len - 1, (self.batch_size,))
        x = torch.stack([self.tokens[i : i + self.seq_len] for i in idx])
        y = torch.stack([self.tokens[i + 1 : i + self.seq_len + 1] for i in idx])
        return x, y


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=int, default=10)
    args = p.parse_args()
    download(args.chunks)
