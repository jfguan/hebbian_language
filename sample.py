"""Generate samples from a trained model checkpoint.

Usage:
    uv run sample.py checkpoints/model_code_memory.pt
    uv run sample.py checkpoints/model_code_memory.pt --prompt "def binary_search("
    uv run sample.py checkpoints/model_code_memory.pt --n 500 --temp 0.9
"""

import argparse
import torch
from models.hebbian_mamba import HebbianMamba


PROMPTS = {
    "code":  'def fizzbuzz(n):\n    """Print 1 to n; Fizz for multiples of 3, Buzz for 5, FizzBuzz for both."""\n',
    "prose": "",
}


@torch.no_grad()
def sample(model, encode, decode, prompt, n, temperature, device):
    model.eval()
    prompt_ids = encode(prompt) if prompt else [0]
    states = None
    for tok_id in prompt_ids[:-1]:
        tok = torch.tensor([tok_id], dtype=torch.long, device=device)
        _, states = model.step(tok, states=states)
        states = [{k: v.detach() if isinstance(v, torch.Tensor) else v
                   for k, v in s.items()} for s in states]
    token = torch.tensor([prompt_ids[-1]], dtype=torch.long, device=device)
    out = []
    for _ in range(n):
        logits, states = model.step(token, states=states)
        states = [{k: v.detach() if isinstance(v, torch.Tensor) else v
                   for k, v in s.items()} for s in states]
        token = torch.multinomial(
            torch.softmax(logits / temperature, dim=-1), 1
        ).squeeze(-1)
        out.append(token.item())
    return prompt + decode(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--prompt", type=str, default=None,
                   help="Custom prompt (overrides default fizzbuzz/prose prompt)")
    p.add_argument("--dataset", type=str, default=None,
                   choices=["code", "prose", "stack"],
                   help="Dataset for tokenizer (inferred from checkpoint name if omitted)")
    p.add_argument("--n",    type=int,   default=400)
    p.add_argument("--temp", type=float, default=0.8)
    args = p.parse_args()

    # Infer dataset from checkpoint name if not specified
    if args.dataset is None:
        name = args.checkpoint.lower()
        if "stack" in name:
            args.dataset = "stack"
        elif "code" in name:
            args.dataset = "code"
        else:
            args.dataset = "prose"

    from data import load_dataset
    dataset_name = "pg19" if args.dataset == "prose" else args.dataset
    ds = load_dataset(dataset_name)

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HebbianMamba(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters())
    cfg = ckpt["config"]
    print(f"Model: {args.checkpoint} ({n_params/1e6:.1f}M d={cfg.d_model} "
          f"L={cfg.n_layers} mem={cfg.use_memory})\n")

    prompt = args.prompt if args.prompt is not None else PROMPTS.get(args.dataset, "")
    print("=" * 60)
    print(sample(model, ds["encode"], ds["decode"], prompt, args.n, args.temp, device))
    print("=" * 60)


if __name__ == "__main__":
    main()
