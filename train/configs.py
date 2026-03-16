from dataclasses import dataclass
from enum import Enum

from data.loader import DatasetName


class ModelType(str, Enum):
    HEBBIAN = "hebbian"
DELTA_HEBBIAN = "delta_hebbian"
    MAMBA = "mamba"
    GDN = "gdn"


@dataclass
class ModelConfig:
    name: str
    model: ModelType
    d_model: int
    n_layers: int
    d_conv: int
    expand: int
    d_state: int
    chunk_size: int

    vocab_size: int = 0  # set from dataset at runtime

    # Hebbian memory
    memory_alpha: float | None = None
    head_dim: int | None = None

    # Delta Hebbian
    delta_layers: str | None = None      # comma-separated layer indices, e.g. "6,7"
    no_memory_layers: str | None = None  # layers with conv+MLP only, no memory
    neg_eigenvalues: bool = False         # beta [0,2] on delta layers

    # GDN
    num_heads: int | None = None


@dataclass
class TrainConfig:
    dataset: DatasetName
    steps: int
    batch_size: int
    seq_len: int
    lr: float
    warmup: int
    grad_accum: int
    eval_interval: int
    ckpt_interval: int


# -- Model configs (name is overridden at runtime to <model>_<dataset>_<size>) --

HEBBIAN_18M = ModelConfig(
    name="hebbian",
    model=ModelType.HEBBIAN,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    memory_alpha=0.2,
)


HEBBIAN_100M = ModelConfig(
    name="hebbian",
    model=ModelType.HEBBIAN,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    memory_alpha=0.2,
)

DELTA_HEBBIAN_18M = ModelConfig(
    name="delta_hebbian",
    model=ModelType.DELTA_HEBBIAN,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    memory_alpha=0.2,
    head_dim=128,
    delta_layers="6,7",
)

DELTA_HEBBIAN_100M = ModelConfig(
    name="delta_hebbian",
    model=ModelType.DELTA_HEBBIAN,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    memory_alpha=0.2,
    head_dim=128,
    delta_layers="10,11",
)

GDN_18M = ModelConfig(
    name="gdn",
    model=ModelType.GDN,
    d_model=512,
    n_layers=6,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=4,
)

MAMBA_18M = ModelConfig(
    name="mamba",
    model=ModelType.MAMBA,
    d_model=512,
    n_layers=10,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
)

MAMBA_100M = ModelConfig(
    name="mamba",
    model=ModelType.MAMBA,
    d_model=1024,
    n_layers=16,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
)


# -- Training hyperparameters --

TRAIN_PG19_18M = TrainConfig(
    dataset=DatasetName.PG19,
    steps=1221,
    batch_size=4,
    seq_len=2048,
    lr=6e-4,
    warmup=60,
    grad_accum=1,
    eval_interval=100,
    ckpt_interval=1221,
)

TRAIN_STACK_18M = TrainConfig(
    dataset=DatasetName.THE_STACK,
    steps=1221,
    batch_size=4,
    seq_len=2048,
    lr=6e-4,
    warmup=60,
    grad_accum=1,
    eval_interval=100,
    ckpt_interval=1221,
)

TRAIN_STACK_100M = TrainConfig(
    dataset=DatasetName.THE_STACK,
    steps=7813,
    batch_size=2,
    seq_len=2048,
    lr=3e-4,
    warmup=500,
    grad_accum=1,
    eval_interval=200,
    ckpt_interval=7813,
)
