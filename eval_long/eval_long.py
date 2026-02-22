"""Long-context loss comparison across models (memory and baselines).

Runs all models on the same val windows at any context length.
W is always updating (normal inference) for memory models.

Usage:
    uv run eval_long/eval_long.py --models model_mem1.pt model_wide.pt model_deep.pt
    uv run eval_long/eval_long.py --models model_mem1.pt model_wide.pt --tokens 65536 --windows 2
"""

import argparse, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import HebbianMamba


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", buffering=1)  # line-buffered: flushes after every \n
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
    def flush(self):
        self.stdout.flush()
        self.f.flush()
    def close(self):
        sys.stdout = self.stdout  # restore before closing so atexit flush doesn't error
        self.f.close()


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = HebbianMamba(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    cfg = ckpt["config"]
    label = f"{path} ({n_params/1e6:.1f}M d={cfg.d_model} L={cfg.n_layers} mem={cfg.use_memory})"
    short = path.replace("model_", "").replace(".pt", "")
    return model, label, short


def detach(states):
    if isinstance(states, torch.Tensor):
        return states.detach()
    if isinstance(states, dict):
        return {k: detach(v) for k, v in states.items()}
    if isinstance(states, (list, tuple)):
        return type(states)(detach(v) for v in states)
    return states


@torch.no_grad()
def run_model(model, tokens, device):
    losses = []
    states = None
    for t in range(len(tokens) - 1):
        tok = torch.tensor([tokens[t]], device=device)
        tgt = torch.tensor([tokens[t + 1]], device=device)
        logits, states = model.step(tok, states=states)
        states = detach(states)
        losses.append(F.cross_entropy(logits, tgt).item())
    return np.array(losses)


def plot(mean_loss, seg_labels, short_names, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = np.arange(len(seg_labels))

    for mi, name in enumerate(short_names):
        ax1.plot(x, mean_loss[mi], marker="o", markersize=3, label=name)

    ax1.set(ylabel="Avg Loss (nats)", title="Per-Segment Loss by Model")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gap vs first model (assumed memory)
    for mi in range(1, len(short_names)):
        gap = mean_loss[mi] - mean_loss[0]
        ax2.plot(x, gap, marker="o", markersize=3, label=f"{short_names[mi]} − {short_names[0]}")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set(xlabel="Segment", ylabel="Loss gap (baseline − memory)", title="Memory Advantage per Segment")
    ax2.set_xticks(x)
    ax2.set_xticklabels(seg_labels, rotation=45, ha="right", fontsize=7)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models",  nargs="+", required=True)
    p.add_argument("--tokens",  type=int, default=4096)
    p.add_argument("--windows", type=int, default=4)
    p.add_argument("--segment", type=int, default=1024)
    p.add_argument("--dataset", type=str, default="pg19", choices=["pg19", "code", "stack"])
    p.add_argument("--out",     type=str, default=None)
    args = p.parse_args()

    if args.dataset == "code":
        from data_code import load_dataset
    elif args.dataset == "stack":
        from data_stack import load_dataset
    else:
        from data import load_dataset

    stem = f"eval_long/{args.tokens}tok_{args.windows}win"
    log_path = args.out or f"{stem}.txt"
    png_path = log_path.replace(".txt", ".png")
    sys.stdout = Tee(log_path)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | {args.tokens} tokens x {args.windows} windows\n")

    ds = load_dataset()
    rng = np.random.default_rng(42)
    max_start = len(ds["val"]) - args.tokens - 1
    starts = sorted(rng.choice(max_start, size=args.windows, replace=False))

    N, seg = args.tokens, args.segment
    n_segs = (N + seg - 1) // seg
    seg_labels = [f"{s//1000}K-{min(s+seg,N)//1000}K" if N >= 4000 else f"{s}-{min(s+seg,N)}"
                  for s in range(0, N, seg)]

    models = [load_model(path, device) for path in args.models]
    short_names = [m[2] for m in models]

    all_losses = np.zeros((len(models), args.windows, n_segs))

    for w_i, start in enumerate(starts):
        tokens = ds["val"][start : start + N + 1].tolist()
        print(f"Window {w_i+1}/{args.windows}: val[{start}:{start+N}]")
        for m_i, (model, label, _) in enumerate(models):
            losses = run_model(model, tokens, device)
            for si, s in enumerate(range(0, N, seg)):
                all_losses[m_i, w_i, si] = losses[s:min(s+seg,N)].mean()
            print(f"  {label}: {losses.mean():.4f}")
        print()

    mean_loss = all_losses.mean(axis=1)
    overall   = all_losses.mean(axis=(1, 2))

    col_w = 10
    print(f"=== Overall (avg over {args.windows} windows) ===")
    print(f"{'Segment':>12}" + "".join(f"{n:>{col_w}}" for n in short_names))
    print("-" * (12 + col_w * len(models)))
    for si, lbl in enumerate(seg_labels):
        print(f"{lbl:>12}" + "".join(f"{mean_loss[mi,si]:>{col_w}.4f}" for mi in range(len(models))))
    print("-" * (12 + col_w * len(models)))
    print(f"{'OVERALL':>12}" + "".join(f"{overall[mi]:>{col_w}.4f}" for mi in range(len(models))))

    print("\nModels:")
    for _, label, _ in models:
        print(f"  {label}")

    plot(mean_loss, seg_labels, short_names, png_path)
    print(f"\nLog: {log_path}")
    sys.stdout.close()


if __name__ == "__main__":
    main()
