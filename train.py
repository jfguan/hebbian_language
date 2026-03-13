"""Unified training script.

Usage:
    uv run train.py configs/hebbian_small.yaml
    uv run train.py configs/hebbian_mamba_100M.yaml --dataset stack --tag my_run
    uv run train.py configs/hebbian_100M.yaml --resume checkpoints/ckpt_hebbian_100M_step2000.pt
    uv run train.py configs/mamba_100M.yaml --compile
"""

import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def build_model(cfg):
    """Instantiate a model from a config dict."""
    model_type = cfg.pop("model")
    if model_type == "hebbian_mamba":
        from models.hebbian_mamba import Config, HebbianMamba
        model_cfg = Config(**{k: v for k, v in cfg.items() if hasattr(Config, k)})
        return HebbianMamba(model_cfg), model_cfg, "HebbianMamba"
    elif model_type == "mamba":
        from models.mamba import Config, Mamba
        model_cfg = Config(**{k: v for k, v in cfg.items() if hasattr(Config, k)})
        return Mamba(model_cfg), model_cfg, "Mamba"
    elif model_type == "hebbian":
        from models.hebbian import Config, HebbianConv
        model_cfg = Config(**{k: v for k, v in cfg.items() if hasattr(Config, k)})
        return HebbianConv(model_cfg), model_cfg, "HebbianConv"
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def cosine_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))


def configure_optimizers(model, lr, weight_decay=0.1):
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in nodecay_params)
    print(f"  decay params: {n_decay:,} | no-decay params: {n_nodecay:,}")
    use_fused = "cuda" in str(next(model.parameters()).device)
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), fused=use_fused)


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
    prompt_ids = encode(prompt) if prompt else [0]
    for tok_id in prompt_ids[:-1]:
        token = torch.tensor([tok_id], dtype=torch.long, device=device)
        _, states = model.step(token, states=states)
        states = [
            {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in s.items()}
            if isinstance(s, dict) else s
            for s in states
        ]
    token = torch.tensor([prompt_ids[-1]], dtype=torch.long, device=device)
    for _ in range(n):
        logits, states = model.step(token, states=states)
        states = [
            {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in s.items()}
            if isinstance(s, dict) else s
            for s in states
        ]
        token = torch.multinomial(
            torch.softmax(logits / temperature, dim=-1), 1
        ).squeeze(-1)
        tokens.append(token.item())
    model.train()
    return prompt + decode(tokens)


def plot_losses(history, tag):
    fig, ax = plt.subplots(figsize=(10, 5))
    use_tokens = "tokens" in history[0]
    xs = [h["tokens"] / 1e6 for h in history] if use_tokens else [h["step"] for h in history]
    xlabel = "Training tokens (M)" if use_tokens else "Step"
    ax.plot(xs, [h["train_loss"] for h in history], label="train", alpha=0.7)
    val = [(h["tokens"] / 1e6 if use_tokens else h["step"], h["val_loss"])
           for h in history if "val_loss" in h]
    if val:
        ax.plot(*zip(*val), "o-", label="val", markersize=4)
    ax.set(xlabel=xlabel, ylabel="loss (nats)", title=f"Training — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"checkpoints/loss_{tag}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="Path to YAML config file")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--compile", action="store_true")
    # Any config value can be overridden from the command line
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--ckpt-interval", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--memory-alpha", type=float, default=None)
    args = p.parse_args()

    # Load config YAML
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Ensure numeric types (YAML may parse scientific notation as strings)
    for k in ("lr", "memory_alpha"):
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = float(cfg[k])

    # CLI overrides
    overrides = {
        "dataset": args.dataset, "tag": args.tag, "steps": args.steps,
        "total_steps": args.total_steps, "batch_size": args.batch_size,
        "seq_len": args.seq_len, "lr": args.lr, "warmup": args.warmup,
        "grad_accum": args.grad_accum, "eval_interval": args.eval_interval,
        "ckpt_interval": args.ckpt_interval, "n_layers": args.n_layers,
        "d_model": args.d_model, "memory_alpha": args.memory_alpha,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    # Extract training params
    tag = cfg.pop("tag")
    dataset = cfg.pop("dataset", "pg19")
    steps = cfg.pop("steps", 1465)
    total_steps = cfg.pop("total_steps", steps)
    batch_size = cfg.pop("batch_size", 2)
    seq_len = cfg.pop("seq_len", 2048)
    lr = cfg.pop("lr", 6e-4)
    warmup = cfg.pop("warmup", 20)
    grad_accum = cfg.pop("grad_accum", 1)
    eval_interval = cfg.pop("eval_interval", 100)
    ckpt_interval = cfg.pop("ckpt_interval", 500)

    # Device
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    device_type = device.split(":")[0]
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Data
    from data import load_dataset, DataLoader
    ds = load_dataset(dataset)
    train_loader = DataLoader(ds["train"], batch_size, seq_len)
    val_loader = DataLoader(ds["val"], batch_size, seq_len)

    # Model
    cfg["vocab_size"] = ds["vocab_size"]
    model_type = cfg["model"]  # keep for printing
    model, model_cfg, model_class = build_model(cfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = configure_optimizers(model, lr)

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        missing, unexpected = raw_model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  New params (randomly initialized): {missing}")
        if unexpected:
            print(f"  Dropped params from checkpoint: {unexpected}")
        if not missing and not unexpected and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    end_step = min(start_step + steps, total_steps)
    min_lr = lr * 0.1
    tokens_per_step = batch_size * seq_len * grad_accum

    print(f"{model_class} | {n_params/1e6:.1f}M params | {device}")
    print(f"Steps {start_step} -> {end_step} (schedule: {total_steps}) | B={batch_size}x{grad_accum} T={seq_len} lr={lr}")

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"histories/history_{tag}.jsonl"
    os.makedirs("histories", exist_ok=True)
    log_file = open(log_path, "a" if args.resume else "w")

    step = start_step
    try:
        for step in range(start_step, end_step):
            t0 = time.time()
            cur_lr = cosine_lr(step, warmup, total_steps, lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(grad_accum):
                x, y = train_loader.batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss = loss / grad_accum
                loss.backward()
                loss_accum += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dt = time.time() - t0
            tokens_seen = (step + 1) * tokens_per_step
            entry = {"step": step, "train_loss": loss_accum, "tokens": tokens_seen}
            print(
                f"step {step:5d} | loss {loss_accum:.4f} | ppl {math.exp(loss_accum):8.2f} | lr {cur_lr:.2e} | {dt*1000:.0f}ms",
                flush=True,
            )

            if step > 0 and step % eval_interval == 0:
                vl = evaluate(model, val_loader, device)
                entry["val_loss"] = vl
                print(f"  val loss {vl:.4f} | val ppl {math.exp(vl):.2f}", flush=True)

            if step > 0 and step % ckpt_interval == 0:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(
                    {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": model_cfg, "model_class": model_class, "step": step},
                    f"checkpoints/ckpt_{tag}_step{step}.pt",
                )
                print(f"  -> checkpoints/ckpt_{tag}_step{step}.pt", flush=True)

            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
    except KeyboardInterrupt:
        print(f"\nStopped at step {step}.")

    # Final save
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    vl = evaluate(model, val_loader, device)
    log_file.write(json.dumps({"step": step, "train_loss": loss_accum, "val_loss": vl, "tokens": (step + 1) * tokens_per_step}) + "\n")
    log_file.close()
    print(f"\nFinal val loss: {vl:.4f} | ppl {math.exp(vl):.2f}")

    if "encode" in ds and "decode" in ds:
        prompt = "def fizzbuzz(n):\n" if dataset in ("code", "stack") else ""
        print(f"Sample:\n{sample(raw_model, ds['encode'], ds['decode'], device, prompt=prompt, n=300)}")

    torch.save(
        {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
         "config": model_cfg, "model_class": model_class, "step": step + 1},
        f"checkpoints/model_{tag}.pt",
    )
    history = [json.loads(line) for line in open(log_path)]
    plot_losses(history, tag)
    print(f"Saved checkpoints/model_{tag}.pt + {log_path}")


if __name__ == "__main__":
    main()
