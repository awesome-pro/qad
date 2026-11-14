"""Evaluate T0 (BF16) or Q0 (W3A16 RTN) on the fixed held-out set.

Teacher-forced NLL and forward KL(teacher || student) over the fixed FineWeb-Edu
blocks. This is the official Gate 2 evaluator; ``--limit-batches`` exists for
local smoke runs and must not be used for reported numbers.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from qwen_qad.losses import (
    chunked_forward_kl,
    chunked_nll,
    gather_scored_tokens,
    next_token_score_mask,
)
from qwen_qad.models import load_model, load_tokenizer, set_trainable_quantized_only
from qwen_qad.quantization import QuantConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", choices=["T0", "Q0"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--eval", default="data/cache/eval/fineweb_eval_ids.npy")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tokenizer = load_tokenizer(args.model, args.revision)
    teacher = load_model(args.model, args.revision, device=args.device, dtype=dtype)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = teacher
    quant_config = None
    if args.variant == "Q0":
        from qwen_qad.models import wrap_quantized_linears

        quant_config = QuantConfig(bits=args.bits, group_size=args.group_size)
        student = load_model(args.model, args.revision, device=args.device, dtype=dtype)
        replaced = wrap_quantized_linears(student, quant_config)
        set_trainable_quantized_only(student)
        for parameter in student.parameters():
            parameter.requires_grad_(False)
        print(f"wrapped {len(replaced)} quantizable linear modules")

    teacher_lm = teacher.get_output_embeddings().weight.detach()
    student_lm = student.get_output_embeddings().weight.detach()

    blocks = np.load(args.eval)
    if args.limit_batches is not None:
        blocks = blocks[: args.batch_size * args.limit_batches]

    total_nll_sum = 0.0
    total_kl_sum = 0.0
    total_tokens = 0
    started = time.perf_counter()

    for start in range(0, blocks.shape[0], args.batch_size):
        input_ids = torch.tensor(
            blocks[start : start + args.batch_size], dtype=torch.long, device=args.device
        )
        seq_len = input_ids.shape[1]
        prompt_lens = torch.ones(input_ids.shape[0], dtype=torch.long, device=args.device)
        response_lens = torch.full_like(prompt_lens, seq_len - 1)
        mask = next_token_score_mask(prompt_lens, response_lens, seq_len, device=args.device)

        with torch.no_grad():
            teacher_hidden = teacher.model(input_ids=input_ids, use_cache=False).last_hidden_state
            h_teacher, targets = gather_scored_tokens(teacher_hidden, input_ids, mask)
            if student is teacher:
                nll = chunked_nll(h_teacher, targets, teacher_lm, chunk_size=args.chunk_size)
                kl = nll.new_zeros(())
            else:
                student_hidden = student.model(
                    input_ids=input_ids, use_cache=False
                ).last_hidden_state
                h_student, _ = gather_scored_tokens(student_hidden, input_ids, mask)
                nll = chunked_nll(h_student, targets, student_lm, chunk_size=args.chunk_size)
                kl = chunked_forward_kl(
                    h_student, h_teacher, student_lm, teacher_lm, chunk_size=args.chunk_size
                )

        count = int(targets.numel())
        total_nll_sum += float(nll.item()) * count
        total_kl_sum += float(kl.item()) * count
        total_tokens += count

    elapsed = time.perf_counter() - started
    nll_mean = total_nll_sum / total_tokens
    kl_mean = total_kl_sum / total_tokens

    peak_memory = None
    if args.device.startswith("cuda"):
        peak_memory = int(torch.cuda.max_memory_allocated())
    elif args.device == "mps":
        peak_memory = int(torch.mps.current_allocated_memory())

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "variant": args.variant,
        "model": args.model,
        "revision": args.revision,
        "tokenizer_revision": args.revision,
        "quantization": quant_config.to_dict() if quant_config else None,
        "eval_set": {
            "path": args.eval,
            "blocks": int(blocks.shape[0]),
            "block_size": int(blocks.shape[1]),
            "tokens_scored": total_tokens,
        },
        "device": args.device,
        "dtype": args.dtype,
        "metrics": {
            "nll": nll_mean,
            "perplexity": float(np.exp(nll_mean)),
            "forward_kl_teacher_student": kl_mean,
        },
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "peak_memory_bytes": peak_memory,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warnings": (
            ["limited batches: not a reportable result"]
            if args.limit_batches is not None
            else []
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
    print(f"tokens={total_tokens:,} elapsed={elapsed:.1f}s "
          f"({result['tokens_per_second']:.0f} tok/s)")
    if peak_memory:
        print(f"peak memory={peak_memory / 2**30:.2f} GiB")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
