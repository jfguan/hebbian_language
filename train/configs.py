from dataclasses import dataclass
from enum import Enum

from data.loader import DatasetName


class ModelType(str, Enum):
    HEBBIAN = "hebbian"
    DELTA_HEBBIAN = "delta_hebbian"
    CONV_ONLY = "conv_only"
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
    num_heads: int | None = None  # memory heads (None or 1 = full D×D)

    # Delta Hebbian
    delta_layers: list[int] | None = None  # layer indices that use delta rule
    delta_num_heads: int | None = None  # heads for delta layers (default: 8)
    no_memory_layers: list[int] | None = None  # layers with conv+MLP only, no memory

    def __post_init__(self):
        # migrate old fields from checkpoints
        for attr in ("head_dim", "delta_head_dim", "memory_alpha", "neg_eigenvalues"):
            if hasattr(self, attr):
                delattr(self, attr)
        if isinstance(self.delta_layers, str):
            self.delta_layers = [int(x) for x in self.delta_layers.split(",")]
        if isinstance(self.no_memory_layers, str):
            self.no_memory_layers = [int(x) for x in self.no_memory_layers.split(",")]


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
    max_steps_per_run: int | None = None


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
    delta_num_heads=2,
    delta_layers=[6, 7],
)

DELTA_HEBBIAN_100M = ModelConfig(
    name="delta_hebbian",
    model=ModelType.DELTA_HEBBIAN,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=32,
    delta_num_heads=8,
    delta_layers=list(range(12)),
)

HYBRID_100M = ModelConfig(
    name="hybrid",
    model=ModelType.DELTA_HEBBIAN,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=1,
    delta_num_heads=8,
    delta_layers=[0, 1, 2, 4, 5, 7, 8, 10, 11],  # regular hebbian at 3, 9
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

GDN_100M = ModelConfig(
    name="gdn",
    model=ModelType.GDN,
    d_model=1024,
    n_layers=9,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=8,
)

CONV_ONLY_18M = ModelConfig(
    name="conv_only",
    model=ModelType.CONV_ONLY,
    d_model=512,
    n_layers=11,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
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
    n_layers=15,
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
    max_steps_per_run=1221,
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
    max_steps_per_run=1221,
)

TRAIN_STACK_100M = TrainConfig(
    dataset=DatasetName.THE_STACK,
    steps=244000,
    batch_size=1,
    seq_len=2048,
    lr=3e-4,
    warmup=5000,
    grad_accum=1,
    eval_interval=1000,
    ckpt_interval=10000,
    max_steps_per_run=48828,
)
