"""Train Conv+Heb on FineWeb (cloud GPU).

Target: ≤3.28 val loss (GPT-2 124M baseline on FineWeb).

Setup on Lambda:
    pip install torch numpy matplotlib huggingface_hub
    python bench_fineweb/data.py
    python bench_fineweb/train_cloud.py
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

from model import Config, HebbianConv


def cosine_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / warmup
    if step >= total:
        return lr_min
    t = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def configure_optimizers(model, lr, weight_decay=0.1):
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embedding" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
    )


@torch.no_grad()
def evaluate(model, val_loader, device, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = val_loader.batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def plot_losses(history, tag):
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = [h["step"] for h in history]
    ax.plot(steps, [h["train_loss"] for h in history], label="train", alpha=0.5)
    val = [(h["step"], h["val_loss"]) for h in history if "val_loss" in h]
    if val:
        ax.plot(*zip(*val), "o-", label="val", markersize=4)
    ax.axhline(y=3.28, color="r", linestyle="--", alpha=0.5, label="GPT-2 target (3.28)")
    ax.set(xlabel="Step", ylabel="Loss (nats)", title=f"Conv+Heb FineWeb — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"bench_fineweb/checkpoints/loss_{tag}.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=19000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--ckpt-interval", type=int, default=2000)
    p.add_argument("--n-layers", type=int, default=18)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--memory-alpha", type=float, default=0.03)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--tag", type=str, default="cloud")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--compile", action="store_true", default=True)
    args = p.parse_args()

    # 32 * 16 * 1024 = 524,288 tokens/step (~10B tokens in 19K steps)

    assert torch.cuda.is_available(), "Cloud training requires CUDA"
    device = "cuda"
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data import load_dataset, DataLoader

    ds = load_dataset()
    train_loader = DataLoader(ds["train"], args.batch_size, args.seq_len)
    val_loader = DataLoader(ds["val"], args.batch_size, args.seq_len)

    cfg = Config(
        vocab_size=ds["vocab_size"],
        d_model=args.d_model,
        d_conv=args.d_conv,
        n_layers=args.n_layers,
        memory_alpha=args.memory_alpha,
    )
    model = HebbianConv(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.1f}M params | {args.n_layers}L d={args.d_model} | {device}")

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = configure_optimizers(model, args.lr)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    total_steps = start_step + args.steps
    min_lr = args.lr * 0.1
    grad_accum = args.grad_accum
    tokens_per_step = args.batch_size * args.seq_len * grad_accum

    print(f"Steps {start_step} -> {total_steps} | B={args.batch_size}x{grad_accum} T={args.seq_len} lr={args.lr}")
    print(f"Tokens/step: {tokens_per_step:,} | Total: {total_steps * tokens_per_step / 1e9:.1f}B tokens")
    print(f"Target: 3.28 val loss")

    ckpt_dir = "bench_fineweb/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = f"{ckpt_dir}/history_{args.tag}.jsonl"
    log_file = open(log_path, "a" if args.resume else "w")

    vl = evaluate(model, val_loader, device)
    print(f"  init val loss {vl:.4f}")

    step = start_step
    entry = {}
    try:
        for step in range(start_step, total_steps):
            t0 = time.time()
            lr = cosine_lr(step, args.warmup, total_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(grad_accum):
                x, y = train_loader.batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss = loss / grad_accum
                loss.backward()
                loss_accum += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dt = time.time() - t0
            tokens_seen = (step + 1) * tokens_per_step
            entry = {"step": step, "train_loss": loss_accum, "tokens": tokens_seen}
            if step % 10 == 0:
                print(
                    f"step {step:5d} | loss {loss_accum:.4f} | lr {lr:.2e}"
                    f" | {dt * 1000:.0f}ms | {tokens_seen / 1e9:.2f}B tok",
                    flush=True,
                )

            if step > 0 and step % args.eval_interval == 0:
                vl = evaluate(model, val_loader, device)
                entry["val_loss"] = vl
                status = "*** REACHED TARGET ***" if vl <= 3.28 else ""
                print(f"  val loss {vl:.4f} {status}", flush=True)

            if step > 0 and step % args.ckpt_interval == 0:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                ckpt_path = f"{ckpt_dir}/ckpt_{args.tag}_step{step}.pt"
                torch.save(
                    {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": cfg,
                        "step": step,
                    },
                    ckpt_path,
                )
                print(f"  -> {ckpt_path}", flush=True)

            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

    except KeyboardInterrupt:
        print(f"\nStopped at step {step}.")

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    vl = evaluate(model, val_loader, device)
    log_file.write(
        json.dumps({"step": step, "train_loss": entry.get("train_loss", 0), "val_loss": vl, "tokens": entry.get("tokens", 0)})
        + "\n"
    )
    log_file.close()
    print(f"\nFinal val loss: {vl:.4f} {'*** REACHED TARGET ***' if vl <= 3.28 else ''}")

    final_path = f"{ckpt_dir}/model_{args.tag}.pt"
    torch.save(
        {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg, "step": step + 1},
        final_path,
    )
    history = [json.loads(line) for line in open(log_path)]
    plot_losses(history, args.tag)
    print(f"Saved {final_path}")


if __name__ == "__main__":
    main()
