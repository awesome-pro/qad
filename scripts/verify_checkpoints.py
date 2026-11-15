"""Verify optimizer checkpoint and portable safetensors encode the same model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_qad.checkpoint import (
    file_sha256,
    import_inference_weights,
    load_checkpoint_verified,
)
from qwen_qad.models import (
    load_model,
    resolve_revisions,
    set_quantization_enabled,
    set_trainable_quantized_only,
    wrap_quantized_linears,
)
from qwen_qad.offline import evaluate_heldout
from qwen_qad.quantization import QuantConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--eval", default="data/cache/eval/fineweb_val_ids.npy")
    parser.add_argument("--eval-blocks", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    revisions = resolve_revisions(args.model, args.revision)
    quant = QuantConfig(bits=args.bits, group_size=args.group_size)
    teacher = load_model(args.model, args.revision, device=device, dtype=dtype)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = load_model(args.model, args.revision, device=device, dtype=dtype)
    wrap_quantized_linears(student, quant)
    set_trainable_quantized_only(student)
    set_quantization_enabled(student, True)
    meta = load_checkpoint_verified(
        args.checkpoint,
        student,
        expected_model_sha=revisions["resolved_model_sha"],
        expected_bits=args.bits,
        expected_group_size=args.group_size,
        expected_scheme="symmetric",
    )

    portable = load_model(args.model, args.revision, device=device, dtype=dtype)
    wrap_quantized_linears(portable, quant)
    set_trainable_quantized_only(portable)
    set_quantization_enabled(portable, True)
    missing = import_inference_weights(args.safetensors, portable)

    blocks = np.load(args.eval)[: args.eval_blocks]
    input_ids = torch.tensor(blocks[:2], dtype=torch.long, device=device)
    with torch.no_grad():
        logits_a = student(input_ids=input_ids, use_cache=False).logits
        logits_b = portable(input_ids=input_ids, use_cache=False).logits
    max_abs = float((logits_a - logits_b).abs().max())
    argmax_match = float((logits_a.argmax(-1) == logits_b.argmax(-1)).float().mean())

    metrics_a = evaluate_heldout(student, teacher, blocks, batch_size=4, chunk_size=256)
    metrics_b = evaluate_heldout(portable, teacher, blocks, batch_size=4, chunk_size=256)
    payload = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "safetensors": args.safetensors,
        "safetensors_sha256": file_sha256(args.safetensors),
        "checkpoint_meta": meta,
        "missing_safetensors_keys": missing,
        "logits": {
            "max_abs_diff": max_abs,
            "argmax_match_fraction": argmax_match,
        },
        "heldout_checkpoint": metrics_a,
        "heldout_safetensors": metrics_b,
        "nll_abs_diff": abs(metrics_a["nll"] - metrics_b["nll"]),
        "kl_abs_diff": abs(
            metrics_a["forward_kl_teacher_student"] - metrics_b["forward_kl_teacher_student"]
        ),
        "passed": bool(
            max_abs < 1e-2
            and argmax_match > 0.999
            and abs(metrics_a["nll"] - metrics_b["nll"]) < 1e-4
            and all(key.startswith("lm_head.") for key in missing)
        ),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("passed", "logits", "nll_abs_diff", "kl_abs_diff")}, indent=2))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
