from dataclasses import dataclass


@dataclass
class ModelConfig:
    model: str
    d_model: int
    n_layers: int
    d_conv: int
    expand: int
    d_state: int
    # Hebbian memory (optional, only for hebbian models)
    memory_alpha: float | None = None
    chunk_size: int | None = None
    # Set at runtime from dataset
    vocab_size: int = 0


@dataclass
class TrainConfig:
    dataset: str
    steps: int
    batch_size: int
    seq_len: int
    lr: float
    warmup: int
    grad_accum: int
    eval_interval: int
    ckpt_interval: int
    compile: bool = False


# -- Hebbian --

HEBBIAN_18M = ModelConfig(
    model="hebbian_minimal",
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    memory_alpha=0.03,
    chunk_size=64,
)

HEBBIAN_100M = ModelConfig(
    model="hebbian_minimal",
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    memory_alpha=0.03,
    chunk_size=64,
)

# -- Hebbian mamba --

HEBBIAN_MAMBA_18M = ModelConfig(
    model="hebbian_mamba",
    d_model=512,
    n_layers=8,
    d_conv=4,
    expand=2,
    d_state=16,
    memory_alpha=0.03,
    chunk_size=64,
)

HEBBIAN_MAMBA_100M = ModelConfig(
    model="hebbian_mamba",
    d_model=1024,
    n_layers=12,
    d_conv=4,
    expand=2,
    d_state=16,
    memory_alpha=0.03,
    chunk_size=64,
)

# -- Mamba baseline --

MAMBA_100M = ModelConfig(
    model="mamba",
    d_model=1024,
    n_layers=16,
    d_conv=4,
    expand=2,
    d_state=16,
)


# -- Training hyperparameters --

TRAIN_18M = TrainConfig(
    dataset="pg19",
    steps=1465,
    batch_size=2,
    seq_len=2048,
    lr=6e-4,
    warmup=20,
    grad_accum=1,
    eval_interval=100,
    ckpt_interval=500,
)

TRAIN_100M = TrainConfig(
    dataset="the_stack",
    steps=64000,
    batch_size=1,
    seq_len=2048,
    lr=3e-4,
    warmup=500,
    grad_accum=1,
    eval_interval=200,
    ckpt_interval=2000,
)
