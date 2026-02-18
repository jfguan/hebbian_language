import argparse
import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import DataLoader, load_dataset
from model import Config, HebbianMamba


def cosine_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    t = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, loader, device, steps=10):
    model.eval()
    total = 0.0
    for _ in range(steps):
        x, y = loader.batch()
        _, loss = model(x.to(device), y.to(device))
        total += loss.item()
    model.train()
    return total / steps


@torch.no_grad()
def sample(model, decode, device, n=200, temperature=0.8):
    model.eval()
    token = torch.zeros(1, dtype=torch.long, device=device)
    states, tokens = None, []
    for _ in range(n):
        logits, states = model.step(token, states=states)
        states = [
            {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in s.items()}
            for s in states
        ]
        token = torch.multinomial(
            torch.softmax(logits / temperature, dim=-1), 1
        ).squeeze(-1)
        tokens.append(token.item())
    model.train()
    return decode(tokens)


def plot_losses(history, tag):
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = [h["step"] for h in history]
    ax.plot(steps, [h["train_loss"] for h in history], label="train", alpha=0.7)
    val = [(h["step"], h["val_loss"]) for h in history if "val_loss" in h]
    if val:
        ax.plot(*zip(*val), "o-", label="val", markersize=4)
    ax.set(xlabel="step", ylabel="loss", title=f"Training — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"loss_{tag}.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--schedule-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    use_memory = not args.no_memory
    tag = args.tag or f"mem{int(use_memory)}"
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    torch.manual_seed(42)

    ds = load_dataset()
    train_loader = DataLoader(ds["train"], args.batch_size, args.seq_len)
    val_loader = DataLoader(ds["val"], args.batch_size, args.seq_len)

    cfg = Config(vocab_size=ds["vocab_size"], use_memory=use_memory, d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state)
    model = HebbianMamba(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  New params (randomly initialized): {missing}")
        if unexpected:
            print(f"  Dropped params from checkpoint: {unexpected}")
        if not missing and not unexpected and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    total_steps = start_step + args.steps
    min_lr = args.lr * 0.1
    T = args.seq_len
    print(
        f"{n_params / 1e6:.1f}M params | memory={use_memory} | {device}"
    )
    print(
        f"Steps {start_step} -> {total_steps} | B={args.batch_size} T={T} lr={args.lr}"
    )

    log_path = f"history_{tag}.jsonl"
    log_file = open(log_path, "a" if args.resume else "w")

    try:
        for step in range(start_step, total_steps):
            t0 = time.time()
            lr = cosine_lr(step, 20, args.schedule_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            x, y = train_loader.batch()
            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            dt = time.time() - t0
            entry = {"step": step, "train_loss": loss.item()}
            print(
                f"step {step:5d} | loss {loss.item():.4f} | ppl {math.exp(loss.item()):8.2f} | lr {lr:.2e} | {dt * 1000:.0f}ms",
                flush=True,
            )

            if step > 0 and step % args.eval_interval == 0:
                vl = evaluate(model, val_loader, device)
                entry["val_loss"] = vl
                print(f"  val loss {vl:.4f} | val ppl {math.exp(vl):.2f}", flush=True)

            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
    except KeyboardInterrupt:
        print(f"\nStopped at step {step}.")

    vl = evaluate(model, val_loader, device)
    log_file.write(
        json.dumps({"step": step, "train_loss": entry["train_loss"], "val_loss": vl})
        + "\n"
    )
    log_file.close()
    print(f"\nFinal val loss: {vl:.4f} | ppl {math.exp(vl):.2f}")
    print(f"Sample:\n{sample(model, ds['decode'], device, 300)}")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "step": step + 1,
        },
        f"model_{tag}.pt",
    )
    history = [json.loads(line) for line in open(log_path)]
    plot_losses(history, tag)
    print(f"Saved model_{tag}.pt + {log_path}")


if __name__ == "__main__":
    main()
