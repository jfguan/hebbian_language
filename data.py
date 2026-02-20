import json
import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
TRAIN_TOKENS_PATH = os.path.join(DATA_DIR, "train_tokens.npy")
VAL_TOKENS_PATH = os.path.join(DATA_DIR, "val_tokens.npy")

# Subsample size for BPE training (chars)
BPE_TRAIN_CHARS = 2_000_000


class BPETokenizer:
    """Minimal byte-level BPE tokenizer. No external dependencies."""

    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.merges = []  # list of (a, b) pairs in merge order
        self.vocab = {}   # id -> bytes

    def train(self, text: str):
        """Learn BPE merges from text."""
        data = list(text.encode("utf-8"))
        # Base vocab: all 256 byte values
        self.vocab = {i: bytes([i]) for i in range(256)}
        num_merges = self.vocab_size - 256

        ids = list(data)
        for i in range(num_merges):
            # Count pairs
            counts = {}
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + i
            self.merges.append(pair)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            # Replace all occurrences of pair in ids
            ids = self._merge(ids, pair, new_id)
            if (i + 1) % 32 == 0:
                print(f"  BPE merge {i + 1}/{num_merges}: {self.vocab[new_id]!r} (freq={counts[pair]})")

    @staticmethod
    def _merge(ids, pair, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for pair_idx, pair in enumerate(self.merges):
            new_id = 256 + pair_idx
            ids = self._merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str):
        obj = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }
        with open(path, "w") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            obj = json.load(f)
        tok = cls(vocab_size=obj["vocab_size"])
        tok.merges = [tuple(p) for p in obj["merges"]]
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(tok.merges):
            tok.vocab[256 + i] = tok.vocab[a] + tok.vocab[b]
        return tok


TRAIN_CHARS_TARGET = 15_000_000  # ~15M chars → ~6M tokens with BPE
VAL_CHARS_TARGET = 1_000_000     # ~1M chars of val text


def _stream_books(split, char_target):
    """Stream books from PG-19 until we hit the char target."""
    from datasets import load_dataset as hf_load

    ds = hf_load("emozilla/pg19", split=split, streaming=True)
    chunks = []
    total = 0
    n_books = 0
    for row in ds:
        chunks.append(row["text"])
        total += len(row["text"])
        n_books += 1
        if n_books % 50 == 0:
            print(f"  {split}: {n_books} books, {total:,} chars...", flush=True)
        if total >= char_target:
            break
    print(f"  {split}: {n_books} books, {total:,} chars (done)")
    return "\n\n".join(chunks)


def _download_pg19():
    """Stream PG-19 books until we have enough text for training."""
    print("Streaming PG-19 train books...")
    train_text = _stream_books("train", TRAIN_CHARS_TARGET)
    print("Streaming PG-19 validation books...")
    val_text = _stream_books("validation", VAL_CHARS_TARGET)
    return train_text, val_text


def load_dataset(vocab_size=512):
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check for cached tokenized arrays
    if os.path.exists(TRAIN_TOKENS_PATH) and os.path.exists(VAL_TOKENS_PATH) and os.path.exists(TOKENIZER_PATH):
        print("Loading cached tokenizer and token arrays...")
        tok = BPETokenizer.load(TOKENIZER_PATH)
        if tok.vocab_size == vocab_size:
            train_data = np.load(TRAIN_TOKENS_PATH)
            val_data = np.load(VAL_TOKENS_PATH)
            print(f"  Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")
            return dict(
                train=train_data,
                val=val_data,
                vocab_size=vocab_size,
                decode=tok.decode,
            )
        print(f"  vocab_size mismatch ({tok.vocab_size} vs {vocab_size}), retraining...")

    # Download PG-19
    train_text, val_text = _download_pg19()

    # Train BPE on a subsample of train text
    bpe_sample = train_text[:BPE_TRAIN_CHARS]
    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(bpe_sample):,} chars...")
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(bpe_sample)
    tok.save(TOKENIZER_PATH)
    print(f"  Saved tokenizer to {TOKENIZER_PATH}")

    # Tokenize and cache
    print("Tokenizing train set (this may take a while)...")
    train_data = np.array(tok.encode(train_text), dtype=np.int64)
    print(f"  {len(train_text):,} chars -> {len(train_data):,} tokens ({len(train_text)/len(train_data):.1f}x compression)")
    np.save(TRAIN_TOKENS_PATH, train_data)

    print("Tokenizing val set...")
    val_data = np.array(tok.encode(val_text), dtype=np.int64)
    print(f"  {len(val_text):,} chars -> {len(val_data):,} tokens ({len(val_text)/len(val_data):.1f}x compression)")
    np.save(VAL_TOKENS_PATH, val_data)

    print(f"Cached token arrays to {DATA_DIR}")
    return dict(
        train=train_data,
        val=val_data,
        vocab_size=vocab_size,
        decode=tok.decode,
    )


class DataLoader:
    def __init__(self, data: np.ndarray, batch_size: int, seq_len: int):
        self.data = data
        self.B = batch_size
        self.T = seq_len

    def batch(self):
        ix = torch.randint(len(self.data) - self.T, (self.B,))
        x = torch.stack([torch.from_numpy(self.data[i : i + self.T].copy()) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.T].copy()) for i in ix])
        return x, y
