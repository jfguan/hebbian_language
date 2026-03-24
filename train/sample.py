"""Generate samples from a trained model checkpoint.

Usage:
    uv run train/sample.py checkpoints/hebbian_18M_the_stack.pt
    uv run train/sample.py checkpoints/hebbian_18M_the_stack.pt --prompt "def binary_search("
    uv run train/sample.py checkpoints/hebbian_18M_the_stack.pt --n 500 --temp 0.9
"""

import argparse

import torch

from data import load_dataset
from data.loader import DatasetName
from models import build_model
from train.run import sample


PROMPTS = {
    DatasetName.THE_STACK: 'def fizzbuzz(n):\n    """Print 1 to n; Fizz for multiples of 3, Buzz for 5, FizzBuzz for both."""\n',
    DatasetName.PG19: "",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--dataset", type=DatasetName, default=None,
                        choices=list(DatasetName),
                        help="Dataset for tokenizer (inferred from checkpoint name if omitted)")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--temp", type=float, default=0.8)
    args = parser.parse_args()

    # infer dataset from checkpoint name
    if args.dataset is None:
        name = args.checkpoint.lower()
        args.dataset = DatasetName.THE_STACK if "stack" in name else DatasetName.PG19

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset(args.dataset)

    # load model from checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = ckpt["model_config"]
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_config.name} ({n_params/1e6:.1f}M d={model_config.d_model} L={model_config.n_layers})")

    prompt = args.prompt if args.prompt is not None else PROMPTS.get(args.dataset, "")
    print("=" * 60)
    print(sample(model, ds.encode, ds.decode, device, prompt=prompt, n=args.n, temperature=args.temp))
    print("=" * 60)


if __name__ == "__main__":
    main()
