"""Unified training script.

Usage:
    uv run train/run.py hebbian_minimal_18M 18M
    uv run train/run.py hebbian_mamba_100M 100M
    uv run train/run.py hebbian_100M 100M --resume checkpoints/ckpt_hebbian_100M_step2000.pt
"""

import argparse
import json
import math
import os
import time
from copy import deepcopy

import torch

from data import load_dataset, DataLoader
from models import build_model
import train.configs as C

MODELS = {
    "hebbian_18M": C.HEBBIAN_18M,
    "hebbian_100M": C.HEBBIAN_100M,
    "hebbian_mamba_18M": C.HEBBIAN_MAMBA_18M,
    "hebbian_mamba_100M": C.HEBBIAN_MAMBA_100M,
    "mamba_100M": C.MAMBA_100M,
}

TRAINS = {
    "train_18M": C.TRAIN_18M,
    "train_100M": C.TRAIN_100M,
}


def main():
    # setup
    args, model_config, train_config, tag = parse_args()
    device = setup_device()
    ds, train_loader, val_loader = setup_data(train_config)
    model_config.vocab_size = ds.vocab_size
    model, checkpoint_config, model_class, optimizer = setup_model(model_config, train_config, device)

    # logging
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("histories", exist_ok=True)
    log_path = f"histories/{tag}.jsonl"
    log_file = open(log_path, "a" if args.resume else "w")

    # print stats
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{model_class} | {n_params / 1e6:.1f}M params | {device}")
    start_step = (
        resume_from(model, optimizer, args.resume, device) if args.resume else 0
    )
    print(
        f"Steps {start_step} -> {train_config.steps} | B={train_config.batch_size}x{train_config.grad_accum} T={train_config.seq_len} lr={train_config.lr}"
    )

    # training loop
    step = start_step
    min_lr = train_config.lr * 0.1
    tokens_per_step = train_config.batch_size * train_config.seq_len * train_config.grad_accum
    for step in range(start_step + 1, train_config.steps + 1):
        t0 = time.time()

        # lr schedule
        cur_lr = cosine_lr(step, train_config.warmup, train_config.steps, train_config.lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # forward + backward with grad accumulation
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_config.grad_accum):
            x, y = train_loader.batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / train_config.grad_accum
            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # log
        entry = {
            "step": step,
            "train_loss": loss_accum,
            "tokens": step * tokens_per_step,
            "lr": cur_lr,
            "dt_ms": (time.time() - t0) * 1000,
        }
        if step % train_config.eval_interval == 0:
            entry["val_loss"] = evaluate(model, val_loader, device)
        log(log_file, entry)

        # Checkpoint
        if step % train_config.ckpt_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                checkpoint_config,
                model_class,
                step,
                f"checkpoints/ckpt_{tag}_step{step}.pt",
            )

    # final log
    entry = {
        "step": step,
        "train_loss": loss_accum,
        "tokens": step * tokens_per_step,
        "val_loss": evaluate(model, val_loader, device),
    }
    log(log_file, entry)
    log_file.close()

    # final checkpoint
    save_checkpoint(
        model, optimizer, checkpoint_config, model_class, step, f"checkpoints/model_{tag}.pt"
    )

    # sample
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    prompt = "def fizzbuzz(n):\n"
    print(
        f"Sample:\n{sample(raw_model, ds.encode, ds.decode, device, prompt=prompt, n=300)}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=MODELS.keys())
    p.add_argument("train", choices=TRAINS.keys())
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)

    args = p.parse_args()
    model_config = deepcopy(MODELS[args.model])
    train_config = deepcopy(TRAINS[args.train])
    tag = args.tag or args.model
    return args, model_config, train_config, tag


def setup_device():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    torch.manual_seed(42)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def setup_data(train_config):
    ds = load_dataset(train_config.dataset)
    train_loader = DataLoader(ds.train, train_config.batch_size, train_config.seq_len)
    val_loader = DataLoader(ds.val, train_config.batch_size, train_config.seq_len)
    return ds, train_loader, val_loader


def setup_model(model_config, train_config, device):
    model, checkpoint_config, model_class = build_model(model_config)
    model = model.to(device)
    if train_config.compile:
        print("Compiling model...")
        model = torch.compile(model)
    optimizer = configure_optimizers(model, train_config.lr)
    return model, checkpoint_config, model_class, optimizer


def configure_optimizers(model, lr, weight_decay=0.1):
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    print(
        f"  decay params: {sum(p.numel() for p in decay_params):,} | no-decay params: {sum(p.numel() for p in nodecay_params):,}"
    )
    use_fused = "cuda" in str(next(model.parameters()).device)
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), fused=use_fused)


def resume_from(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    missing, unexpected = raw_model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  New params (randomly initialized): {missing}")
    if unexpected:
        print(f"  Dropped params from checkpoint: {unexpected}")
    if not missing and not unexpected and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt.get("step", 0)
    print(f"Resumed from {path} at step {start_step}")
    return start_step


def cosine_lr(step, warmup, total, max_lr, min_lr):
    if step <= warmup:
        return max_lr * step / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))


def log(log_file, entry):
    print(
        " | ".join(
            f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
            for k, v in entry.items()
        ),
        flush=True,
    )
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


@torch.no_grad()
def evaluate(model, loader, device, steps=10):
    model.eval()
    total = 0.0
    for _ in range(steps):
        x, y = loader.batch()
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    model.train()
    return total / steps


def save_checkpoint(model, optimizer, checkpoint_config, model_class, step, path):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": checkpoint_config,
            "model_class": model_class,
            "step": step,
        },
        path,
    )
    print(f"  -> {path}", flush=True)


@torch.no_grad()
def sample(model, encode, decode, device, prompt="", n=200, temperature=0.8):
    model.eval()
    states, tokens = None, []

    # process prompt
    prompt_ids = encode(prompt) if prompt else [0]
    for tok_id in prompt_ids[:-1]:
        token = torch.tensor([tok_id], dtype=torch.long, device=device)
        _, states = model.step(token, states=states)
        states = detach_states(states)

    # generate
    token = torch.tensor([prompt_ids[-1]], dtype=torch.long, device=device)
    for _ in range(n):
        logits, states = model.step(token, states=states)
        states = detach_states(states)
        token = torch.multinomial(
            torch.softmax(logits / temperature, dim=-1), 1
        ).squeeze(-1)
        tokens.append(token.item())
    model.train()
    return prompt + decode(tokens)


def detach_states(states):
    return [
        {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in s.items()}
        if isinstance(s, dict)
        else s
        for s in states
    ]


if __name__ == "__main__":
    main()
