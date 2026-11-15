"""Evaluate T0 (BF16) or Q0 (W3A16 RTN) on the fixed held-out set.

Teacher-forced NLL and forward KL(teacher || student) over the fixed FineWeb-Edu
blocks. For T0 the KL is computed numerically (not assigned), and a two-pass
teacher self-KL is measured on a few batches as a determinism check.

``--limit-batches`` exists for local smoke runs and must not be used for
reported numbers.
"""

from __future__ import annotations

import argparse
import hashlib
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
from qwen_qad.memory import MemorySampler, parameter_bytes
from qwen_qad.models import (
    load_model,
    load_tokenizer,
    resolve_revisions,
    set_trainable_quantized_only,
)
from qwen_qad.quantization import QuantConfig
from qwen_qad.checkpoint import file_sha256, load_checkpoint_verified


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
    parser.add_argument("--self-kl-batches", type=int, default=4)
    parser.add_argument("--checkpoint", default=None)
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
    device = torch.device(args.device)

    revisions = resolve_revisions(args.model, args.revision)
    tokenizer = load_tokenizer(args.model, args.revision)
    teacher = load_model(args.model, args.revision, device=device, dtype=dtype)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = teacher
    quant_config = None
    checkpoint_info = None
    if args.variant == "Q0":
        from qwen_qad.models import wrap_quantized_linears

        quant_config = QuantConfig(bits=args.bits, group_size=args.group_size)
        student = load_model(args.model, args.revision, device=device, dtype=dtype)
        replaced = wrap_quantized_linears(student, quant_config)
        set_trainable_quantized_only(student)
        for parameter in student.parameters():
            parameter.requires_grad_(False)
        if args.checkpoint:
            checkpoint_sha256 = file_sha256(args.checkpoint)
            meta = load_checkpoint_verified(
                args.checkpoint,
                student,
                expected_model_sha=revisions["resolved_model_sha"],
                expected_bits=args.bits,
                expected_group_size=args.group_size,
                expected_scheme="symmetric",
            )
            checkpoint_info = {
                "path": args.checkpoint,
                "sha256": checkpoint_sha256,
                "meta": meta,
            }
        print(f"wrapped {len(replaced)} quantizable linear modules")

    teacher_lm = teacher.get_output_embeddings().weight.detach()
    student_lm = student.get_output_embeddings().weight.detach()

    blocks = np.load(args.eval)
    full_blocks = blocks.shape[0]
    if args.limit_batches is not None:
        blocks = blocks[: args.batch_size * args.limit_batches]

    total_nll_sum = 0.0
    total_kl_sum = 0.0
    total_tokens = 0
    self_kl_sum = 0.0
    self_kl_max = 0.0
    self_kl_tokens = 0
    self_kl_batches = 0
    started = time.perf_counter()

    with MemorySampler(device) as sampler:
        for batch_index, start in enumerate(range(0, blocks.shape[0], args.batch_size)):
            input_ids = torch.tensor(
                blocks[start : start + args.batch_size], dtype=torch.long, device=device
            )
            seq_len = input_ids.shape[1]
            prompt_lens = torch.ones(input_ids.shape[0], dtype=torch.long, device=device)
            response_lens = torch.full_like(prompt_lens, seq_len - 1)
            mask = next_token_score_mask(prompt_lens, response_lens, seq_len, device=device)

            with torch.no_grad():
                teacher_hidden = teacher.model(
                    input_ids=input_ids, use_cache=False
                ).last_hidden_state
                h_teacher, targets = gather_scored_tokens(teacher_hidden, input_ids, mask)

                if student is teacher:
                    nll = chunked_nll(h_teacher, targets, teacher_lm, chunk_size=args.chunk_size)
                    # Computed, not assigned. With the same tensors this is exactly 0;
                    # the two-pass check below measures the real numerical self-KL.
                    kl = chunked_forward_kl(
                        h_teacher, h_teacher, teacher_lm, teacher_lm, chunk_size=args.chunk_size
                    )
                    if batch_index < args.self_kl_batches:
                        second_hidden = teacher.model(
                            input_ids=input_ids, use_cache=False
                        ).last_hidden_state
                        h_second, second_targets = gather_scored_tokens(
                            second_hidden, input_ids, mask
                        )
                        two_pass = chunked_forward_kl(
                            h_second, h_teacher, teacher_lm, teacher_lm,
                            chunk_size=args.chunk_size,
                        )
                        self_kl_sum += float(two_pass) * second_targets.numel()
                        self_kl_max = max(self_kl_max, abs(float(two_pass)))
                        self_kl_tokens += int(second_targets.numel())
                        self_kl_batches += 1
                else:
                    student_hidden = student.model(
                        input_ids=input_ids, use_cache=False
                    ).last_hidden_state
                    h_student, _ = gather_scored_tokens(student_hidden, input_ids, mask)
                    nll = chunked_nll(
                        h_student, targets, student_lm, chunk_size=args.chunk_size
                    )
                    kl = chunked_forward_kl(
                        h_student, h_teacher, student_lm, teacher_lm,
                        chunk_size=args.chunk_size,
                    )

            count = int(targets.numel())
            total_nll_sum += float(nll.item()) * count
            total_kl_sum += float(kl.item()) * count
            total_tokens += count

    elapsed = time.perf_counter() - started
    nll_mean = total_nll_sum / total_tokens
    kl_mean = total_kl_sum / total_tokens

    memory = sampler.result()
    memory["parameter_bytes"] = {
        "teacher": parameter_bytes(teacher),
        "student": parameter_bytes(student),
        "note": (
            "teacher and student are the same object for T0"
            if student is teacher
            else "teacher and quantized student are separate objects"
        ),
    }

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "variant": args.variant,
        "model": args.model,
        "revisions": revisions,
        "quantization": quant_config.to_dict() if quant_config else None,
        "checkpoint": checkpoint_info,
        "eval_set": {
            "path": args.eval,
            "blocks_used": int(blocks.shape[0]),
            "blocks_available": int(full_blocks),
            "block_size": int(blocks.shape[1]),
            "tokens_scored": total_tokens,
            "ids_sha256": hashlib.sha256(blocks.tobytes()).hexdigest(),
            "limited": args.limit_batches is not None,
        },
        "device": str(device),
        "dtype": args.dtype,
        "metrics": {
            "nll": nll_mean,
            "perplexity": float(np.exp(nll_mean)),
            "forward_kl_teacher_student": kl_mean,
        },
        "teacher_self_kl": (
            {
                "computed": True,
                "two_pass_mean": self_kl_sum / self_kl_tokens if self_kl_tokens else None,
                "two_pass_max_batch": self_kl_max if self_kl_batches else None,
                "batches": self_kl_batches,
                "tokens": self_kl_tokens,
                "note": "computed between two independent forward passes on the same batch",
            }
            if student is teacher
            else None
        ),
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "memory": memory,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warnings": (
            ["limited batches: not a reportable result"] if args.limit_batches is not None else []
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
    if result["teacher_self_kl"]:
        print(f"teacher self-KL (two pass): {result['teacher_self_kl']['two_pass_mean']:.3e}")
    print(f"tokens={total_tokens:,} elapsed={elapsed:.1f}s "
          f"({result['tokens_per_second']:.0f} tok/s)")
    print(f"memory: {memory['peak_bytes']} bytes peak ({memory['peak_method']})")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
