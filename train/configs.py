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


# -- Hebbian --

HEBBIAN_18M = ModelConfig(
    name="hebbian_18M",
    model=ModelType.HEBBIAN,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    memory_alpha=0.03,
)

HEBBIAN_512_18M = ModelConfig(
    name="hebbian_512_18M",
    model=ModelType.HEBBIAN,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    memory_alpha=0.03,
)

HEBBIAN_BD256_18M = ModelConfig(
    name="hebbian_bd256_18M",
    model=ModelType.HEBBIAN,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    memory_alpha=0.03,
    head_dim=256,
)


HEBBIAN_100M = ModelConfig(
    name="hebbian_100M",
    model=ModelType.HEBBIAN,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    memory_alpha=0.03,
)

# -- Delta Hebbian --

DELTA_HEBBIAN_18M = ModelConfig(
    name="delta_hebbian_18M",
    model=ModelType.DELTA_HEBBIAN,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    memory_alpha=0.03,
    head_dim=256,
)

DELTA_HEBBIAN_100M = ModelConfig(
    name="delta_hebbian_100M",
    model=ModelType.DELTA_HEBBIAN,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    memory_alpha=0.03,
    head_dim=256,
)

# -- GDN baseline --

GDN_18M = ModelConfig(
    name="gdn_18M",
    model=ModelType.GDN,
    d_model=512,
    n_layers=6,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=4,
)

# -- Mamba baseline --

MAMBA_18M = ModelConfig(
    name="mamba_18M",
    model=ModelType.MAMBA,
    d_model=512,
    n_layers=10,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
)

MAMBA_100M = ModelConfig(
    name="mamba_100M",
    model=ModelType.MAMBA,
    d_model=1024,
    n_layers=16,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
)


# -- Training hyperparameters --

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
