"""Code dataset loader — StarCoderData Python, filtered for long files.

Streams bigcode/starcoderdata (python subset), keeps files >= MIN_FILE_CHARS,
trains a BPE tokenizer on a subsample, and caches token arrays as .npy files.
"""

import os

import numpy as np
import torch

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
TRAIN_TOKENS_PATH = os.path.join(DATA_DIR, "train_tokens.npy")
VAL_TOKENS_PATH = os.path.join(DATA_DIR, "val_tokens.npy")

MIN_FILE_CHARS = 4096     # only keep long files
TRAIN_CHARS_TARGET = 15_000_000
VAL_CHARS_TARGET = 1_000_000
BPE_TRAIN_CHARS = 2_000_000

# Re-use the BPETokenizer from data/
import sys
sys.path.insert(0, os.path.dirname(DATA_DIR))
from data import BPETokenizer, DataLoader  # noqa: E402


def _stream_code(char_target, seed=None):
    """Stream Python files from codeparrot-clean until char_target is reached."""
    from datasets import load_dataset as hf_load

    ds = hf_load("codeparrot/codeparrot-clean", split="train", streaming=True)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)

    chunks = []
    total = 0
    n_files = 0
    for row in ds:
        content = row["content"]
        if len(content) < MIN_FILE_CHARS:
            continue
        chunks.append(content)
        total += len(content)
        n_files += 1
        if n_files % 200 == 0:
            print(f"  {n_files} files, {total:,} chars...", flush=True)
        if total >= char_target:
            break
    print(f"  {n_files} files, {total:,} chars (done)")
    return "\n\n".join(chunks)


def load_dataset(vocab_size=512):
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(TRAIN_TOKENS_PATH) and os.path.exists(VAL_TOKENS_PATH) and os.path.exists(TOKENIZER_PATH):
        print("Loading cached code tokenizer and token arrays...")
        tok = BPETokenizer.load(TOKENIZER_PATH)
        if tok.vocab_size == vocab_size:
            train_data = np.load(TRAIN_TOKENS_PATH)
            val_data = np.load(VAL_TOKENS_PATH)
            print(f"  Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")
            return dict(train=train_data, val=val_data, vocab_size=vocab_size, decode=tok.decode)
        print(f"  vocab_size mismatch ({tok.vocab_size} vs {vocab_size}), retraining...")

    print("Streaming StarCoderData Python (train)...")
    train_text = _stream_code(TRAIN_CHARS_TARGET, seed=42)
    print("Streaming StarCoderData Python (val)...")
    val_text = _stream_code(VAL_CHARS_TARGET, seed=1337)

    bpe_sample = train_text[:BPE_TRAIN_CHARS]
    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(bpe_sample):,} chars...")
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(bpe_sample)
    tok.save(TOKENIZER_PATH)
    print(f"  Saved tokenizer to {TOKENIZER_PATH}")

    print("Tokenizing train set...")
    train_data = np.array(tok.encode(train_text), dtype=np.int64)
    print(f"  {len(train_text):,} chars -> {len(train_data):,} tokens ({len(train_text)/len(train_data):.1f}x compression)")
    np.save(TRAIN_TOKENS_PATH, train_data)

    print("Tokenizing val set...")
    val_data = np.array(tok.encode(val_text), dtype=np.int64)
    print(f"  {len(val_text):,} chars -> {len(val_data):,} tokens ({len(val_text)/len(val_data):.1f}x compression)")
    np.save(VAL_TOKENS_PATH, val_data)

    print(f"Cached token arrays to {DATA_DIR}")
    return dict(train=train_data, val=val_data, vocab_size=vocab_size, decode=tok.decode)
