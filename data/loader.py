from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial

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
    stream_train: Callable[[int], str]  # (char_target) -> text
    stream_val: Callable[[int], str]


def _stream_pg19(char_target: int, hf_split: str) -> str:
    ds = hf_load("emozilla/pg19", split=hf_split, streaming=True)
    texts = (row["text"] for row in ds)

    return _collect_chunks(texts, char_target)


def _stream_stack(char_target: int, seed: int) -> str:
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

    # Filter for only files above 32_000 chararacter length
    texts = (row["content"] for row in combined if len(row["content"]) >= 32_000)
    return _collect_chunks(texts, char_target)


def _collect_chunks(texts: Iterator[str], char_target: int) -> str:
    """Accumulate text chunks from an iterator until char_target is reached."""
    chunks, char_total = [], 0
    while char_total < char_target:
        text = next(texts)

        chunks.append(text)
        char_total += len(text)

        if len(chunks) % 100 == 0:
            print(f"  {len(chunks)} items, {char_total:,} chars...", flush=True)

    print(f"  {len(chunks)} items, {char_total:,} chars (done)")
    return "\n\n".join(chunks)


DATA_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS: dict[str, DatasetConfig] = {
    "pg19": DatasetConfig(
        cache_dir=os.path.join(DATA_DIR, "pg19"),
        vocab_size=1024,
        train_chars=80_000_000,
        val_chars=4_000_000,
        bpe_train_chars=5_000_000,
        stream_train=partial(_stream_pg19, hf_split="train"),
        stream_val=partial(_stream_pg19, hf_split="validation"),
    ),
    "the_stack": DatasetConfig(
        cache_dir=os.path.join(DATA_DIR, "the_stack"),
        vocab_size=1024,
        train_chars=64_000_000,
        val_chars=4_000_000,
        bpe_train_chars=5_000_000,
        stream_train=partial(_stream_stack, seed=42),
        stream_val=partial(_stream_stack, seed=1337),
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
    """Load (or download + tokenize + cache) a dataset."""
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
        print(
            f"vocab_size mismatch ({tokenizer.get_vocab_size()} vs {cfg.vocab_size}), retraining"
        )
        return None

    train_data = np.load(train_path)
    val_data = np.load(val_path)
    return Dataset(train_data, val_data, cfg.vocab_size, tokenizer)


def _download_dataset(cfg: DatasetConfig) -> Dataset:
    # Stream text from HuggingFace
    train_text = cfg.stream_train(cfg.train_chars)
    val_text = cfg.stream_val(cfg.val_chars)

    # Train BPE tokenizer on subset of training text
    tokenizer_path = os.path.join(cfg.cache_dir, "tokenizer.json")
    bpe_sample = train_text[: cfg.bpe_train_chars]
    print(
        f"Training BPE tokenizer (vocab_size={cfg.vocab_size}) on {len(bpe_sample):,} chars..."
    )

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=cfg.vocab_size, special_tokens=[])
    # Split into lines so BPE sees natural boundaries
    tokenizer.train_from_iterator(bpe_sample.splitlines(), trainer=trainer)
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
