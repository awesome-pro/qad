"""Gate 1: load the checkpoint, emit the authoritative parameter inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwen_qad.models import build_inventory, load_model, load_tokenizer, summarize_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    tokenizer = load_tokenizer(args.model, args.revision)
    model = load_model(args.model, args.revision, device=args.device, dtype=dtype)
    inventory = build_inventory(model, tokenizer, args.model, args.revision)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, indent=2) + "\n")

    print(summarize_inventory(inventory))
    print(f"\nwritten to {out}")

    modules = inventory["linear_modules"]["modules"]
    print(f"\nfirst 8 of {len(modules)} Linear modules:")
    for module in modules[:8]:
        print(f"  {module['name']:55s} {module['shape']} quantizable={module['quantizable']}")


if __name__ == "__main__":
    main()
