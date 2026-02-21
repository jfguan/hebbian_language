import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import Config, HebbianMamba


def cosine_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    t = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))


def configure_optimizers(model, lr, weight_decay):
    """Separate weight decay for 2D params only (nanoGPT style)."""
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
    # feed prompt tokens through model to build state
    for tok_id in prompt_ids[:-1]:
        token = torch.tensor([tok_id], dtype=torch.long, device=device)
        _, states = model.step(token, states=states)
        states = [
            {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in s.items()}
            for s in states
        ]
    token = torch.tensor([prompt_ids[-1]], dtype=torch.long, device=device)
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
    return prompt + decode(tokens)


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
    fig.savefig(f"checkpoints/loss_{tag}.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="pg19", choices=["pg19", "code"])
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--dual-memory", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--steps", type=int, default=1465)
    p.add_argument("--schedule-steps", type=int, default=1465)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--ckpt-interval", type=int, default=500)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    use_memory = not args.no_memory
    dual_memory = args.dual_memory
    dataset_prefix = "code_" if args.dataset == "code" else ""
    tag = args.tag or (
        f"{dataset_prefix}memory_dual" if dual_memory
        else f"{dataset_prefix}memory" if use_memory
        else f"{dataset_prefix}no_memory"
    )
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

    if args.dataset == "code":
        from data_code import load_dataset, DataLoader
    else:
        from data import load_dataset, DataLoader
    ds = load_dataset()
    train_loader = DataLoader(ds["train"], args.batch_size, args.seq_len)
    val_loader = DataLoader(ds["val"], args.batch_size, args.seq_len)

    cfg = Config(vocab_size=ds["vocab_size"], use_memory=use_memory, dual_memory=dual_memory, d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state)
    model = HebbianMamba(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    optimizer = configure_optimizers(model, args.lr, weight_decay=0.1)

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

    total_steps = start_step + args.steps
    min_lr = args.lr * 0.1
    T = args.seq_len
    grad_accum = args.grad_accum
    print(
        f"{n_params / 1e6:.1f}M params | memory={use_memory} | {device}"
    )
    print(
        f"Steps {start_step} -> {total_steps} | B={args.batch_size}x{grad_accum} T={T} lr={args.lr}"
    )

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"checkpoints/history_{tag}.jsonl"
    log_file = open(log_path, "a" if args.resume else "w")

    try:
        for step in range(start_step, total_steps):
            t0 = time.time()
            lr = cosine_lr(step, 20, args.schedule_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for micro in range(grad_accum):
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
            entry = {"step": step, "train_loss": loss_accum}
            print(
                f"step {step:5d} | loss {loss_accum:.4f} | ppl {math.exp(loss_accum):8.2f} | lr {lr:.2e} | {dt * 1000:.0f}ms",
                flush=True,
            )

            if step > 0 and step % args.eval_interval == 0:
                vl = evaluate(model, val_loader, device)
                entry["val_loss"] = vl
                print(f"  val loss {vl:.4f} | val ppl {math.exp(vl):.2f}", flush=True)

            if step > 0 and step % args.ckpt_interval == 0:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                            "config": cfg, "step": step},
                           f"checkpoints/ckpt_{tag}_step{step}.pt")
                print(f"  -> checkpoints/ckpt_{tag}_step{step}.pt", flush=True)

            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
    except KeyboardInterrupt:
        print(f"\nStopped at step {step}.")

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    vl = evaluate(model, val_loader, device)
    log_file.write(
        json.dumps({"step": step, "train_loss": entry["train_loss"], "val_loss": vl})
        + "\n"
    )
    log_file.close()
    print(f"\nFinal val loss: {vl:.4f} | ppl {math.exp(vl):.2f}")
    prompt = "def fizzbuzz(n):\n" if args.dataset == "code" else ""
    print(f"Sample:\n{sample(raw_model, ds['encode'], ds['decode'], device, prompt=prompt, n=300)}")

    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "step": step + 1,
        },
        f"checkpoints/model_{tag}.pt",
    )
    history = [json.loads(line) for line in open(log_path)]
    plot_losses(history, tag)
    print(f"Saved checkpoints/model_{tag}.pt + {log_path}")


if __name__ == "__main__":
    main()
