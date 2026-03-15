"""Unified training script.

Usage:
    uv run train/run.py train/models/hebbian_minimal_18M.yaml train/configs/config_18M.yaml
    uv run train/run.py train/models/hebbian_mamba_100M.yaml train/configs/config_100M.yaml
    uv run train/run.py train/models/hebbian_100M.yaml train/configs/config_100M.yaml --resume checkpoints/ckpt_hebbian_100M_step2000.pt
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from models import build_model


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


def main():
    args, model_cfg_dict, tc, tag = parse_args()

    # Set Device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    device_type = device.split(":")[0]
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # data
    from data import load_dataset, DataLoader

    ds = load_dataset(tc.dataset)
    train_loader = DataLoader(ds.train, tc.batch_size, tc.seq_len)
    val_loader = DataLoader(ds.val, tc.batch_size, tc.seq_len)

    # model
    model_cfg_dict["vocab_size"] = ds.vocab_size
    model, model_cfg, model_class = build_model(model_cfg_dict)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = configure_optimizers(model, tc.lr)

    # resume
    start_step = 0
    if args.resume:
        start_step = resume_from(model, optimizer, args.resume, device)

    # training loop
    end_step = tc.steps
    min_lr = tc.lr * 0.1
    tokens_per_step = tc.batch_size * tc.seq_len * tc.grad_accum

    print(f"{model_class} | {n_params / 1e6:.1f}M params | {device}")
    print(
        f"Steps {start_step} -> {end_step} | B={tc.batch_size}x{tc.grad_accum} T={tc.seq_len} lr={tc.lr}"
    )

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"histories/history_{tag}.jsonl"
    os.makedirs("histories", exist_ok=True)
    log_file = open(log_path, "a" if args.resume else "w")

    step = start_step
    try:
        for step in range(start_step, end_step):
            t0 = time.time()

            # lr schedule
            cur_lr = cosine_lr(step, tc.warmup, tc.steps, tc.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            # forward + backward with grad accumulation
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(tc.grad_accum):
                x, y = train_loader.batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss = loss / tc.grad_accum
                loss.backward()
                loss_accum += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # logging
            dt = time.time() - t0
            tokens_seen = (step + 1) * tokens_per_step
            entry = {"step": step, "train_loss": loss_accum, "tokens": tokens_seen}
            print(
                f"step {step:5d} | loss {loss_accum:.4f} | ppl {math.exp(loss_accum):8.2f} | lr {cur_lr:.2e} | {dt * 1000:.0f}ms",
                flush=True,
            )

            # eval
            if step > 0 and step % tc.eval_interval == 0:
                vl = evaluate(model, val_loader, device)
                entry["val_loss"] = vl
                print(f"  val loss {vl:.4f} | val ppl {math.exp(vl):.2f}", flush=True)

            # checkpoint
            if step > 0 and step % tc.ckpt_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    model_cfg,
                    model_class,
                    step,
                    f"checkpoints/ckpt_{tag}_step{step}.pt",
                )

            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
    except KeyboardInterrupt:
        print(f"\nStopped at step {step}.")

    # final save
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    vl = evaluate(model, val_loader, device)
    log_file.write(
        json.dumps(
            {
                "step": step,
                "train_loss": loss_accum,
                "val_loss": vl,
                "tokens": (step + 1) * tokens_per_step,
            }
        )
        + "\n"
    )
    log_file.close()
    print(f"\nFinal val loss: {vl:.4f} | ppl {math.exp(vl):.2f}")

    prompt = "def fizzbuzz(n):\n" if tc.dataset in ("code_parrot", "the_stack") else ""
    print(
        f"Sample:\n{sample(raw_model, ds.encode, ds.decode, device, prompt=prompt, n=300)}"
    )

    save_checkpoint(
        model,
        optimizer,
        model_cfg,
        model_class,
        step + 1,
        f"checkpoints/model_{tag}.pt",
    )
    history = [json.loads(line) for line in open(log_path)]
    plot_losses(history, tag)
    print(f"Saved checkpoints/model_{tag}.pt + {log_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model_config", type=str, help="Path to model YAML config")
    p.add_argument("train_config", type=str, help="Path to training YAML config")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    for k in ("lr", "memory_alpha"):
        if k in model_cfg and isinstance(model_cfg[k], str):
            model_cfg[k] = float(model_cfg[k])

    with open(args.train_config) as f:
        train_dict = yaml.safe_load(f)

    tc = TrainConfig(**{k: v for k, v in train_dict.items() if hasattr(TrainConfig, k)})
    tag = args.tag or os.path.splitext(os.path.basename(args.model_config))[0]
    return args, model_cfg, tc, tag



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


def save_checkpoint(model, optimizer, model_cfg, model_class, step, path):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model_cfg,
            "model_class": model_class,
            "step": step,
        },
        path,
    )
    print(f"  -> {path}", flush=True)


def cosine_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))


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


def plot_losses(history, tag):
    use_tokens = "tokens" in history[0]
    xs = (
        [h["tokens"] / 1e6 for h in history]
        if use_tokens
        else [h["step"] for h in history]
    )
    xlabel = "Training tokens (M)" if use_tokens else "Step"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, [h["train_loss"] for h in history], label="train", alpha=0.7)
    val = [
        (h["tokens"] / 1e6 if use_tokens else h["step"], h["val_loss"])
        for h in history
        if "val_loss" in h
    ]
    if val:
        ax.plot(*zip(*val), "o-", label="val", markersize=4)
    ax.set(xlabel=xlabel, ylabel="loss (nats)", title=f"Training — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"checkpoints/loss_{tag}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
