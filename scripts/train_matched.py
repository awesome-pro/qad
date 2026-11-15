"""Matched offline/on-policy QAD trainer (Priority 1).

Both arms use the same prompt schedule, prompt IDs per step, continuation cap,
steps, optimizer, LR schedule, seed, teacher and quantization. The only
intended difference is the continuation source: gold for Q1, current-student
rollouts for Q2.

Records per-step prompt hashes, token counts, generation/training time, prompt
coverage, validation curve, and best/final checkpoints with hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from qwen_qad.checkpoint import export_inference_weights, save_checkpoint
from qwen_qad.memory import MemorySampler, parameter_bytes
from qwen_qad.models import (
    load_model,
    load_tokenizer,
    resolve_revisions,
    set_quantization_enabled,
    set_trainable_quantized_only,
    wrap_quantized_linears,
)
from qwen_qad.offline import Example, evaluate_heldout, example_to_batch, forward_kl_step
from qwen_qad.quantization import FakeQuantLinear, QuantConfig
from qwen_qad.rollouts import generate_rollouts
from qwen_qad.schedule import (
    build_schedule,
    load_prompt_records,
    step_indices,
    step_prompt_hashes,
)


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["q1", "q2"], required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--prompts", default="data/cache/prompts/openthoughts_prompts.jsonl")
    parser.add_argument("--val-set", default="data/cache/eval/fineweb_val_ids.npy")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=32)
    parser.add_argument("--continuation-cap", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-blocks", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out-dir", default="experiments/gate8")
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
    records = load_prompt_records(args.prompts)
    schedule = build_schedule(records, args.max_seq_len, args.continuation_cap, args.seed)

    teacher = load_model(args.model, args.revision, device=device, dtype=dtype)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = load_model(args.model, args.revision, device=device, dtype=dtype)
    quant_config = QuantConfig(bits=args.bits, group_size=args.group_size)
    wrapped = wrap_quantized_linears(student, quant_config)
    trainable_params = set_trainable_quantized_only(student)
    set_quantization_enabled(student, True)

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda(step, args.warmup_steps, args.steps)
    )
    val_blocks = np.load(args.val_set)[: args.eval_blocks * args.eval_batch_size]
    val_hash = hashlib.sha256(val_blocks.tobytes()).hexdigest()

    validation = []
    steps_log = []
    best_metric = math.inf
    best_step = None
    best_checkpoint = None
    final_checkpoint = None
    seen_prompts: set[int] = set()
    started = time.perf_counter()
    total_generation_seconds = 0.0
    aborted = None

    with MemorySampler(device) as sampler:
        initial = evaluate_heldout(student, teacher, val_blocks, args.eval_batch_size, args.chunk_size)
        validation.append({"step": 0, "lr": 0.0, **initial})
        print(f"step 0 validation: {initial}")

        for step in range(1, args.steps + 1):
            step_started = time.perf_counter()
            indices = step_indices(schedule, step, args.prompts_per_step)
            seen_prompts.update(indices)
            hashes = step_prompt_hashes(records, indices)
            generation_seconds = 0.0

            if args.arm == "q1":
                examples = [
                    Example(
                        prompt_ids=[int(t) for t in records[i]["prompt_ids"]],
                        response_ids=[
                            int(t) for t in records[i]["response_ids"][: args.continuation_cap]
                        ],
                        index=i,
                    )
                    for i in indices
                ]
            else:
                prompt_ids_list = [
                    [int(t) for t in records[i]["prompt_ids"]] for i in indices
                ]
                for module in student.modules():
                    if isinstance(module, FakeQuantLinear):
                        module.invalidate_cache()
                generation_started = time.perf_counter()
                rollouts = generate_rollouts(
                    student,
                    tokenizer,
                    prompt_ids_list,
                    max_new_tokens=args.continuation_cap,
                    batch_size=args.generation_batch_size,
                    seed=args.seed + step * 1000,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
                generation_seconds = time.perf_counter() - generation_started
                total_generation_seconds += generation_seconds
                examples = [
                    Example(
                        prompt_ids=rollout["prompt_ids"],
                        response_ids=rollout["response_ids"][: args.continuation_cap],
                        index=rollout["index"],
                    )
                    for rollout in rollouts
                    if rollout["response_len"] >= 2
                ]

            student.train()
            optimizer.zero_grad(set_to_none=True)
            total_tokens = sum(len(example.response_ids) for example in examples)
            if total_tokens < 1:
                aborted = f"no usable tokens at step {step}"
                break
            loss_sum = 0.0
            for example in examples:
                ids, mask, _, _ = example_to_batch(example, device)
                loss, n = forward_kl_step(student, teacher, ids, mask, chunk_size=args.chunk_size)
                if not torch.isfinite(loss):
                    aborted = f"non-finite loss at step {step}"
                    break
                (loss * (n / total_tokens)).backward()
                loss_sum += float(loss.detach()) * n
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
                "loss": loss_sum / total_tokens,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "prompt_instances": len(indices),
                "usable_continuations": len(examples),
                "response_tokens": total_tokens,
                "prompt_hashes": hashes,
                "generation_seconds": generation_seconds,
                "student_seconds": time.perf_counter() - step_started - generation_seconds,
                "total_seconds": time.perf_counter() - step_started,
            }

            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate_heldout(
                    student, teacher, val_blocks, args.eval_batch_size, args.chunk_size
                )
                validation.append({"step": step, "lr": record["lr"], **metrics})
                record["val_nll"] = metrics["nll"]
                if metrics["nll"] < best_metric:
                    best_metric = metrics["nll"]
                    best_step = step
                    best_path = ckpt_dir / f"{args.run_name}_best.pt"
                    meta = {
                        "arm": args.arm,
                        "run_name": args.run_name,
                        "step": step,
                        "val_nll": metrics["nll"],
                        "revisions": revisions,
                        "quantization": quant_config.to_dict(),
                        "schedule_seed": args.seed,
                        "prompts_per_step": args.prompts_per_step,
                        "continuation_cap": args.continuation_cap,
                    }
                    size = save_checkpoint(best_path, student, optimizer, step, meta)
                    inference_path = ckpt_dir / f"{args.run_name}_best.safetensors"
                    export_inference_weights(
                        inference_path,
                        student,
                        metadata={
                            "arm": args.arm,
                            "step": str(step),
                            "val_nll": str(metrics["nll"]),
                            "model": args.model,
                            "revision": str(revisions.get("resolved_model_sha")),
                            "quantization": json.dumps(quant_config.to_dict()),
                        },
                    )
                    best_checkpoint = {
                        "path": str(best_path),
                        "step": step,
                        "val_nll": metrics["nll"],
                        "bytes": size,
                        "sha256": file_sha256(best_path),
                        "inference_weights": str(inference_path),
                        "inference_bytes": inference_path.stat().st_size,
                        "inference_sha256": file_sha256(inference_path),
                    }
                print(f"step {step} val: {metrics}")

            steps_log.append(record)
            print(
                f"step {step}: loss={record['loss']:.5f} lr={record['lr']:.2e} "
                f"grad={record['grad_norm']:.2f} tokens={total_tokens} "
                f"gen={generation_seconds:.1f}s total={record['total_seconds']:.1f}s"
            )

    final_path = ckpt_dir / f"{args.run_name}_final.pt"
    if not aborted:
        meta = {
            "arm": args.arm,
            "run_name": args.run_name,
            "step": args.steps,
            "revisions": revisions,
            "quantization": quant_config.to_dict(),
            "schedule_seed": args.seed,
            "prompts_per_step": args.prompts_per_step,
            "continuation_cap": args.continuation_cap,
        }
        size = save_checkpoint(final_path, student, optimizer, args.steps, meta)
        inference_path = ckpt_dir / f"{args.run_name}_final.safetensors"
        export_inference_weights(
            inference_path,
            student,
            metadata={
                "arm": args.arm,
                "step": str(args.steps),
                "model": args.model,
                "revision": str(revisions.get("resolved_model_sha")),
                "quantization": json.dumps(quant_config.to_dict()),
            },
        )
        final_checkpoint = {
            "path": str(final_path),
            "step": args.steps,
            "bytes": size,
            "sha256": file_sha256(final_path),
            "inference_weights": str(inference_path),
            "inference_bytes": inference_path.stat().st_size,
            "inference_sha256": file_sha256(inference_path),
        }

    elapsed = time.perf_counter() - started
    teacher_grads = [p.grad for p in teacher.parameters() if p.grad is not None]
    trainable_grads = [p.grad for p in trainable if p.grad is not None]
    quantized_active = sum(
        1 for m in student.modules() if isinstance(m, FakeQuantLinear) and m.quantized
    )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "arm": args.arm,
        "run_name": args.run_name,
        "model": {"id": args.model, "revisions": revisions},
        "quantization": quant_config.to_dict(),
        "trainable_parameters": trainable_params,
        "wrapped_modules": len(wrapped),
        "schedule": {
            "seed": args.seed,
            "prompts_per_step": args.prompts_per_step,
            "continuation_cap": args.continuation_cap,
            "eligible_prompts": len(schedule),
            "unique_prompts_seen": len(seen_prompts),
            "prompt_instances": sum(s["prompt_instances"] for s in steps_log),
        },
        "data": {
            "prompts": args.prompts,
            "val_set": args.val_set,
            "val_sha256": val_hash,
            "val_blocks": int(val_blocks.shape[0]),
        },
        "optimizer": {
            "name": "AdamW", "lr": args.lr, "betas": [0.9, 0.95],
            "weight_decay": 0.0, "warmup_steps": args.warmup_steps,
            "steps": args.steps,
        },
        "generation": (
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "batch_size": args.generation_batch_size,
                "seed_rule": "schedule_seed + step*1000 + batch_start",
            }
            if args.arm == "q2"
            else None
        ),
        "validation_curve": validation,
        "steps_log": steps_log,
        "best_checkpoint": best_checkpoint,
        "final_checkpoint": final_checkpoint,
        "timing": {
            "elapsed_seconds": elapsed,
            "generation_seconds": total_generation_seconds,
            "response_tokens": sum(s["response_tokens"] for s in steps_log),
        },
        "memory": sampler.result(),
        "parameter_bytes": {
            "teacher": parameter_bytes(teacher), "student": parameter_bytes(student)
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
            "platform": platform.platform(),
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out_dir / f"{args.run_name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out_dir / f"{args.run_name}_steps.jsonl").open("w") as handle:
        for record in steps_log:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(payload["assertions"], indent=2))
    print(f"best: {best_checkpoint}")
    print(f"final: {final_checkpoint}")


if __name__ == "__main__":
    main()
