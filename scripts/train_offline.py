"""Offline QAD (Q1): teacher-student forward KL on fixed gold completions.

Pilot trainer for the W4A16-G128 pipeline-verification arm. A short run
(default 25 steps) is a probe, not a full training run. No CE term, no
on-policy rollouts, no benchmark generation.

Key invariants checked and recorded:
- teacher parameters frozen, no grads;
- only the wrapped linear masters are trainable;
- quantization active in train and eval;
- gradients finite, loss finite;
- checkpoint saved, keys match, one master tensor matches after reload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from qwen_qad.checkpoint import save_checkpoint
from qwen_qad.memory import MemorySampler, parameter_bytes
from qwen_qad.models import (
    load_model,
    load_tokenizer,
    resolve_revisions,
    set_quantization_enabled,
    set_trainable_quantized_only,
    wrap_quantized_linears,
)
from qwen_qad.offline import (
    build_step_batch,
    evaluate_heldout,
    example_to_batch,
    forward_kl_step,
    make_example,
)
from qwen_qad.quantization import FakeQuantLinear, QuantConfig


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--prompts", default="data/cache/prompts/openthoughts_prompts.jsonl")
    parser.add_argument("--eval", default="data/cache/eval/fineweb_eval_ids.npy")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--response-tokens-per-step", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--response-cap", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-blocks", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="q1_w4a16_g128_probe")
    parser.add_argument("--out-dir", default="experiments/gate3")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    revisions = resolve_revisions(args.model, args.revision)
    tokenizer = load_tokenizer(args.model, args.revision)

    teacher = load_model(args.model, args.revision, device=device, dtype=dtype)
    for p in teacher.parameters():
        p.requires_grad_(False)

    quant_config = QuantConfig(bits=args.bits, group_size=args.group_size)
    student = load_model(args.model, args.revision, device=device, dtype=dtype)
    wrapped = wrap_quantized_linears(student, quant_config)
    trainable_params = set_trainable_quantized_only(student)
    set_quantization_enabled(student, True)
    student.train()

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda(step, args.warmup_steps, args.steps)
    )

    records = []
    with Path(args.prompts).open() as handle:
        for line in handle:
            records.append(json.loads(line))
    examples = []
    for index, record in enumerate(records):
        example = make_example(record, args.max_seq_len, index, args.response_cap)
        if example is not None:
            examples.append(example)
    if not examples:
        raise RuntimeError("no usable examples")
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    prompt_hash = hashlib.sha256(
        json.dumps([e.prompt_ids for e in examples], sort_keys=True).encode()
    ).hexdigest()

    blocks = np.load(args.eval)[: args.eval_blocks * args.eval_batch_size]
    eval_hash = hashlib.sha256(blocks.tobytes()).hexdigest()

    validation_curve = []
    steps_log = []
    cursor = 0
    started = time.perf_counter()
    aborted = None

    with MemorySampler(device) as sampler:
        initial = evaluate_heldout(
            student, teacher, blocks, args.eval_batch_size, args.chunk_size
        )
        validation_curve.append({"step": 0, "lr": 0.0, **initial})
        print(f"step 0 (Q0) validation: {initial}")

        for step in range(1, args.steps + 1):
            batch, cursor, tokens = build_step_batch(
                examples, cursor, args.response_tokens_per_step
            )
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            loss_sum = 0.0
            micro_tokens = 0
            for example in batch:
                ids, mask, prompt_len, response_len = example_to_batch(example, device)
                loss, n = forward_kl_step(
                    student, teacher, ids, mask, chunk_size=args.chunk_size
                )
                if not torch.isfinite(loss):
                    aborted = f"non-finite loss at step {step}"
                    break
                scaled = loss * (n / tokens)
                scaled.backward()
                loss_sum += float(loss.detach()) * n
                micro_tokens += n
            if aborted:
                break
            grad_norm = torch.sqrt(
                sum(
                    (p.grad.detach().float() ** 2).sum()
                    for p in trainable
                    if p.grad is not None
                )
            )
            if not torch.isfinite(grad_norm):
                aborted = f"non-finite gradient at step {step}"
                break
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()
            for module in student.modules():
                if isinstance(module, FakeQuantLinear):
                    module.invalidate_cache()

            record = {
                "step": step,
                "loss": loss_sum / tokens,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "examples": len(batch),
                "response_tokens": tokens,
                "seconds": time.perf_counter() - step_started,
            }
            steps_log.append(record)
            print(
                f"step {step}: loss={record['loss']:.5f} lr={record['lr']:.2e} "
                f"grad={record['grad_norm']:.2f} examples={len(batch)} "
                f"tokens={tokens} {record['seconds']:.2f}s"
            )
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate_heldout(
                    student, teacher, blocks, args.eval_batch_size, args.chunk_size
                )
                validation_curve.append({"step": step, "lr": record["lr"], **metrics})
                print(f"step {step} validation: {metrics}")

    elapsed = time.perf_counter() - started
    checkpoint_path = ckpt_dir / f"{args.run_name}_step{args.steps}.pt"
    checkpoint_bytes = None
    checkpoint_check: dict[str, object] = {"saved": False}
    if not aborted:
        meta = {
            "run_name": args.run_name,
            "model": args.model,
            "revisions": revisions,
            "quantization": quant_config.to_dict(),
            "data": {"prompts_sha256": prompt_hash, "eval_sha256": eval_hash},
            "steps": args.steps,
            "trainable_parameters": trainable_params,
        }
        checkpoint_bytes = save_checkpoint(checkpoint_path, student, optimizer, args.steps, meta)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sample_key = next(
            name for name in payload["model"] if name.endswith("q_proj.weight")
        )
        current = student.state_dict()[sample_key].cpu()
        reloaded = payload["model"][sample_key]
        checkpoint_check = {
            "saved": True,
            "path": str(checkpoint_path),
            "bytes": checkpoint_bytes,
            "keys_match": set(payload["model"]) == set(student.state_dict()),
            "sample_key": sample_key,
            "sample_max_abs_diff": float((current - reloaded).abs().max()),
            "optimizer_state_present": payload.get("optimizer") is not None,
        }

    teacher_grads = [p.grad for p in teacher.parameters() if p.grad is not None]
    trainable_grads = [p.grad for p in trainable if p.grad is not None]
    quantized_active = sum(
        1 for m in student.modules() if isinstance(m, FakeQuantLinear) and m.quantized
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "run_name": args.run_name,
        "model": {"id": args.model, "revisions": revisions},
        "quantization": quant_config.to_dict(),
        "mode": "offline_qad_forward_kl",
        "trainable_parameters": trainable_params,
        "wrapped_modules": len(wrapped),
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "scheduler": "linear warmup + cosine to 10%",
        },
        "data": {
            "prompts": args.prompts,
            "prompts_sha256": prompt_hash,
            "examples": len(examples),
            "eval": args.eval,
            "eval_blocks": int(blocks.shape[0]),
            "eval_sha256": eval_hash,
        },
        "train_config": {
            "steps": args.steps,
            "response_tokens_per_step": args.response_tokens_per_step,
            "max_seq_len": args.max_seq_len,
            "warmup_steps": args.warmup_steps,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "validation_curve": validation_curve,
        "steps_log": steps_log,
        "elapsed_seconds": elapsed,
        "response_tokens_seen": sum(s["response_tokens"] for s in steps_log),
        "checkpoint": checkpoint_check,
        "memory": sampler.result(),
        "parameter_bytes": {
            "teacher": parameter_bytes(teacher),
            "student": parameter_bytes(student),
        },
        "assertions": {
            "teacher_gradients_absent": len(teacher_grads) == 0,
            "student_gradients_finite": all(
                g is None or bool(torch.isfinite(g).all()) for g in trainable_grads
            ),
            "quantized_forward_active": quantized_active == len(wrapped),
            "loss_finite": all(math.isfinite(s["loss"]) for s in steps_log),
            "aborted": aborted,
        },
        "environment": {
            "device": str(device),
            "dtype": args.dtype,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warnings": (["limited run: probe only"] if args.steps <= 50 else []),
    }
    (out_dir / f"{args.run_name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out_dir / f"{args.run_name}_steps.jsonl").open("w") as handle:
        for record in steps_log:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(payload["assertions"], indent=2))
    print(f"checkpoint: {checkpoint_check}")
    print(f"written {out_dir / (args.run_name + '.json')}")


if __name__ == "__main__":
    main()
