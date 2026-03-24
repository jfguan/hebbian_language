"""Sleep consolidation: two-pass KL alignment to transfer memory → MLP.

Usage:
    uv run experiments/sleep.py hebbian_18M train_stack_18M --steps 500
"""

import argparse
import time
from contextlib import contextmanager
from dataclasses import replace

import torch
import torch.nn.functional as F

from data import load_dataset, DataLoader
from models import build_model
from train.run import MODELS, TRAINS, setup_device, evaluate


@contextmanager
def memory_disabled(model):
    """Temporarily replace each layer's memory with identity."""
    saved = []
    for layer in model.layers:
        if hasattr(layer, "memory"):
            saved.append((layer, layer.memory))
            layer.memory = torch.nn.Identity()
    yield
    for layer, mem in saved:
        layer.memory = mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS.keys())
    parser.add_argument("train", choices=TRAINS.keys())
    args = parser.parse_args()

    device = setup_device()
    torch.manual_seed(42)

    # load model from checkpoint
    model_config = replace(MODELS[args.model])
    train_config = replace(TRAINS[args.train])
    dataset = load_dataset(train_config.dataset)
    model_config.vocab_size = dataset.vocab_size
    model = build_model(model_config).to(device)

    checkpoint_path = f"checkpoints/{args.model}_{train_config.dataset.value}.pt"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    train_loader = DataLoader(dataset.train, train_config.batch_size, train_config.seq_len)
    val_loader = DataLoader(dataset.val, train_config.batch_size, train_config.seq_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr * 0.01)

    # baseline
    val_with = evaluate(model, val_loader, device)
    with memory_disabled(model):
        val_without = evaluate(model, val_loader, device)
    print(f"baseline | with={val_with:.4f} without={val_without:.4f} gap={val_without - val_with:.4f}")

    # consolidation
    T = 4.0
    steps = train_config.steps
    for step in range(1, steps + 1):
        t0 = time.time()
        x, _ = train_loader.batch()
        x = x.to(device)

        # teacher: with memory
        model.eval()
        with torch.no_grad():
            teacher_logits, _ = model(x)

        # student: without memory
        model.train()
        with memory_disabled(model):
            student_logits, _ = model(x)

        loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            vw = evaluate(model, val_loader, device)
            with memory_disabled(model):
                vwo = evaluate(model, val_loader, device)
            print(f"step {step} | kl={loss.item():.4f} | with={vw:.4f} without={vwo:.4f} gap={vwo - vw:.4f} | {(time.time() - t0) * 1000:.0f}ms")


if __name__ == "__main__":
    main()
