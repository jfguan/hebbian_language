from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset as hf_load, interleave_datasets
import numpy as np
import numpy.typing as npt
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


@dataclass
class DatasetConfig:
    cache_dir: str
    vocab_size: int
    train_chars: int
    val_chars: int
    bpe_train_chars: int
    stream: Callable[[int, str, int], str]  # (char_target, split, seed) -> text


def _stream_pg19(char_target: int, split: str, seed: int) -> str:
    # PG-19 uses "validation" not "val" on HuggingFace
    hf_split = "validation" if split == "val" else split
    ds = hf_load("emozilla/pg19", split=hf_split, streaming=True)

    texts = (row["text"] for row in ds)
    return _collect_chunks(texts, char_target)


def _stream_code(char_target: int, split: str, seed: int) -> str:
    ds = hf_load("codeparrot/codeparrot-clean", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    texts = _filter_by_length(ds, "content", min_chars=4096)
    return _collect_chunks(texts, char_target)


def _stream_stack(char_target: int, split: str, seed: int) -> str:
    streams = []
    languages = ["python", "javascript", "typescript", "java", "c", "cpp", "rust", "go"]
    for lang in languages:
        ds = hf_load(
            "bigcode/the-stack-dedup",
            data_dir=f"data/{lang}",
            split="train",
            streaming=True,
        )
        streams.append(ds.select_columns(["content"]))

    combined = interleave_datasets(streams, seed=seed)
    texts = _filter_by_length(combined, "content", min_chars=32_000)
    return _collect_chunks(texts, char_target)


def _collect_chunks(texts: Iterator[str], char_target: int) -> str:
    """Accumulate text chunks from an iterator until char_target is reached."""
    chunks, total, n = [], 0, 0
    for text in texts:
        chunks.append(text)
        total += len(text)
        n += 1
        if n % 100 == 0:
            print(f"  {n} items, {total:,} chars...", flush=True)
        if total >= char_target:
            break
    print(f"  {n} items, {total:,} chars (done)")
    return "\n\n".join(chunks)


def _filter_by_length(dataset: Any, field: str, min_chars: int) -> Iterator[str]:
    """Yield text values from dataset rows that meet the minimum length."""
    for row in dataset:
        if len(row[field]) >= min_chars:
            yield row[field]


DATA_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS: dict[str, DatasetConfig] = {
    "pg19": DatasetConfig(
        cache_dir=os.path.join(DATA_DIR, "pg19"),
        vocab_size=1024,
        train_chars=80_000_000,
        val_chars=4_000_000,
        bpe_train_chars=5_000_000,
        stream=_stream_pg19,
    ),
    "code_parrot": DatasetConfig(
        cache_dir=os.path.join(DATA_DIR, "codeparrot"),
        vocab_size=1024,
        train_chars=64_000_000,
        val_chars=4_000_000,
        bpe_train_chars=5_000_000,
        stream=_stream_code,
    ),
    "the_stack": DatasetConfig(
        cache_dir=os.path.join(DATA_DIR, "the_stack"),
        vocab_size=1024,
        train_chars=64_000_000,
        val_chars=4_000_000,
        bpe_train_chars=5_000_000,
        stream=_stream_stack,
    ),
}


class Dataset:
    """Result of load_dataset: holds token arrays, vocab size, and encode/decode methods."""

    def __init__(
        self,
        train: npt.NDArray[np.uint16],
        val: npt.NDArray[np.uint16],
        vocab_size: int,
        tokenizer: Tokenizer,
    ) -> None:
        self.train = train
        self.val = val
        self.vocab_size = vocab_size
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)


def load_dataset(name: str = "pg19") -> Dataset:
    """Load (or download + tokenize + cache) a dataset.

    name: One of "pg19", "code_parrot", "the_stack".
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from {list(DATASETS)}")

    cfg = DATASETS[name]
    os.makedirs(cfg.cache_dir, exist_ok=True)

    # Load cached dataset, download otherwise.
    return _load_cached_dataset(cfg, name) or _download_dataset(cfg)


def _load_cached_dataset(cfg: DatasetConfig, name: str) -> Dataset | None:
    tokenizer_path = os.path.join(cfg.cache_dir, "tokenizer.json")
    train_path = os.path.join(cfg.cache_dir, "train_tokens.npy")
    val_path = os.path.join(cfg.cache_dir, "val_tokens.npy")

    # Check all cached files exist
    if not (
        os.path.exists(tokenizer_path)
        and os.path.exists(train_path)
        and os.path.exists(val_path)
    ):
        return None

    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except Exception:
        print("Incompatible tokenizer cache, retraining")
        return None

    if tokenizer.get_vocab_size() != cfg.vocab_size:
        print(f"vocab_size mismatch ({tokenizer.get_vocab_size()} vs {cfg.vocab_size}), retraining")
        return None

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    return Dataset(train_data, val_data, cfg.vocab_size, tokenizer)


def _download_dataset(cfg: DatasetConfig) -> Dataset:
    # Stream text from HuggingFace
    train_text = cfg.stream(cfg.train_chars, "train", seed=42)
    val_text = cfg.stream(cfg.val_chars, "val", seed=1337)

    # Train BPE tokenizer on subset of training text
    tokenizer_path = os.path.join(cfg.cache_dir, "tokenizer.json")
    bpe_sample = train_text[: cfg.bpe_train_chars]
    print(f"Training BPE tokenizer (vocab_size={cfg.vocab_size}) on {len(bpe_sample):,} chars...")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=cfg.vocab_size, special_tokens=[])
    tokenizer.train_from_iterator([bpe_sample], trainer=trainer)
    tokenizer.save(tokenizer_path)

    print(f"Saved tokenizer to {tokenizer_path}")

    # Tokenize, save to disk
    train_data = _tokenize(tokenizer, train_text, "train")
    np.save(os.path.join(cfg.cache_dir, "train_tokens.npy"), train_data)

    val_data = _tokenize(tokenizer, val_text, "val")
    np.save(os.path.join(cfg.cache_dir, "val_tokens.npy"), val_data)

    print(f"Cached token arrays to {cfg.cache_dir}")

    return Dataset(train_data, val_data, cfg.vocab_size, tokenizer)


def _tokenize(tokenizer: Tokenizer, text: str, label: str) -> npt.NDArray[np.uint16]:
    print(f"Tokenizing {label} set...")
    data = np.array(tokenizer.encode(text).ids, dtype=np.uint16)

    n_chars, n_tokens = len(text), len(data)
    ratio = n_chars / n_tokens
    print(f"  {n_chars:,} chars -> {n_tokens:,} tokens ({ratio:.1f}x compression)")
    return data


class DataLoader:
    """Simple random-batch data loader from a flat token array."""

    def __init__(
        self, data: npt.NDArray[np.uint16], batch_size: int, seq_len: int
    ) -> None:
        self.data = data
        self.B = batch_size
        self.T = seq_len

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.T, (self.B,))
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + self.T].copy()) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(self.data[i + 1 : i + 1 + self.T].copy()) for i in ix]
        )
        return x, y
