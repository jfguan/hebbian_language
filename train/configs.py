from dataclasses import dataclass
from enum import Enum

from data.loader import DatasetName


class ModelType(str, Enum):
    HEBBIAN = "hebbian"
    DELTA_HEBBIAN = "delta_hebbian"
    DUAL_DELTA = "dual_delta"
    SWA_DELTA = "swa_delta"
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

    # SWA
    swa_window: int = 256  # sliding window attention size

    def __post_init__(self):
        # migrate old fields from checkpoints
        for attr in (
            "head_dim",
            "delta_head_dim",
            "memory_alpha",
            "neg_eigenvalues",
            "no_memory_layers",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        if isinstance(self.delta_layers, str):
            self.delta_layers = [int(x) for x in self.delta_layers.split(",")]


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
    compile: bool = False


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


DUAL_DELTA_18M = ModelConfig(
    name="dual_delta",
    model=ModelType.DUAL_DELTA,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=4,
    delta_num_heads=4,
    delta_layers=[3, 7],  # 6 SWA + 2 dual delta (3:1 ratio)
)

DUAL_DELTA_100M = ModelConfig(
    name="dual_delta",
    model=ModelType.DUAL_DELTA,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=8,
    delta_num_heads=8,
    delta_layers=[3, 7, 11],  # 9 SWA + 3 dual delta
)

SWA_DELTA_18M = ModelConfig(
    name="swa_delta",
    model=ModelType.SWA_DELTA,
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=4,
    delta_num_heads=4,
    delta_layers=[6, 7],  # 6 SWA + 2 delta at the end
)

SWA_DELTA_100M = ModelConfig(
    name="swa_delta",
    model=ModelType.SWA_DELTA,
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    chunk_size=64,
    num_heads=8,
    delta_num_heads=8,
    delta_layers=[3, 7, 11],  # 9 SWA + 3 delta
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
