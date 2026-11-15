"""Clean-process verification that a backed-up .pt and safetensors agree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"]
    portable = load_file(args.safetensors)
    missing = sorted(set(portable) - set(state))
    unexpected = sorted(set(state) - set(portable))
    max_abs = 0.0
    mismatched = []
    for key, tensor in portable.items():
        if key not in state:
            continue
        diff = float((state[key].float() - tensor.float()).abs().max())
        max_abs = max(max_abs, diff)
        if diff != 0.0:
            mismatched.append({"key": key, "max_abs_diff": diff})
    result = {
        "checkpoint": args.checkpoint,
        "safetensors": args.safetensors,
        "checkpoint_step": payload.get("step"),
        "checkpoint_meta": payload.get("meta"),
        "safetensors_tensor_count": len(portable),
        "checkpoint_key_count": len(state),
        "missing_in_checkpoint": missing,
        "unexpected_in_checkpoint": unexpected,
        "compared_tensors": len(portable) - len(missing),
        "max_abs_diff": max_abs,
        "mismatched_tensors": mismatched,
        "passed": bool(
            not missing
            and all(key.startswith("lm_head.") for key in unexpected)
            and max_abs == 0.0
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "compared_tensors", "max_abs_diff", "missing_in_checkpoint")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
