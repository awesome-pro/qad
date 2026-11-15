"""On-policy QAD (Q2): forward KL on student-generated rollouts.

Each step samples fresh rollouts from the current student, scores teacher and
student on the identical token sequence, recomputes the student forward with
autograd, and updates. This is true on-policy data collection with the same
forward-KL objective as Q1, isolating the prefix source.

Short run by default; W4A16-G128 pipeline verification.
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
    Example,
    evaluate_heldout,
    example_to_batch,
    forward_kl_step,
)
from qwen_qad.quantization import FakeQuantLinear, QuantConfig
from qwen_qad.rollouts import generate_rollouts


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
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-blocks", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="q2_w4a16_g128_200")
    parser.add_argument("--out-dir", default="experiments/gate4")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--compare-q1", default="experiments/gate3/q1_w4a16_g128_200.json")
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

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_lambda(step, args.warmup_steps, args.steps)
    )

    prompts = []
    with Path(args.prompts).open() as handle:
        for line in handle:
            record = json.loads(line)
            prompt_ids = [int(t) for t in record["prompt_ids"]]
            if len(prompt_ids) + 8 <= args.max_seq_len:
                prompts.append(prompt_ids)
    if len(prompts) < args.prompts_per_step:
        raise RuntimeError("not enough prompts")
    rng = random.Random(args.seed)
    rng.shuffle(prompts)
    prompt_hash = hashlib.sha256(
        json.dumps(prompts, sort_keys=True).encode()
    ).hexdigest()

    blocks = np.load(args.eval)[: args.eval_blocks * args.eval_batch_size]
    eval_hash = hashlib.sha256(blocks.tobytes()).hexdigest()

    q1_tokens = None
    if Path(args.compare_q1).exists():
        q1_tokens = json.loads(Path(args.compare_q1).read_text()).get("response_tokens_seen")

    validation_curve = []
    steps_log = []
    started = time.perf_counter()
    rollout_hashes = []
    aborted = None

    with MemorySampler(device) as sampler:
        initial = evaluate_heldout(student, teacher, blocks, args.eval_batch_size, args.chunk_size)
        validation_curve.append({"step": 0, "lr": 0.0, **initial})
        print(f"step 0 (Q0) validation: {initial}")

        for step in range(1, args.steps + 1):
            step_started = time.perf_counter()
            start = ((step - 1) * args.prompts_per_step) % len(prompts)
            selected = [prompts[(start + i) % len(prompts)] for i in range(args.prompts_per_step)]
            for module in student.modules():
                if isinstance(module, FakeQuantLinear):
                    module.invalidate_cache()
            generation_started = time.perf_counter()
            rollouts = generate_rollouts(
                student,
                tokenizer,
                selected,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.generation_batch_size,
                seed=args.seed + step * 1000,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
            generation_seconds = time.perf_counter() - generation_started
            student.train()

            usable = [r for r in rollouts if r["response_len"] >= 2]
            if not usable:
                aborted = f"no usable rollouts at step {step}"
                break
            total_tokens = sum(r["response_len"] for r in usable)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for rollout in usable:
                example = Example(
                    prompt_ids=rollout["prompt_ids"],
                    response_ids=rollout["response_ids"],
                    index=rollout["index"],
                )
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

            rollout_hash = hashlib.sha256(
                json.dumps([r["response_ids"] for r in usable]).encode()
            ).hexdigest()[:16]
            rollout_hashes.append(rollout_hash)
            record = {
                "step": step,
                "loss": loss_sum / total_tokens,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "rollouts": len(usable),
                "response_tokens": total_tokens,
                "mean_response_len": total_tokens / len(usable),
                "eos_rate": sum(r["eos_generated"] for r in usable) / len(usable),
                "mean_repetition_4gram": sum(r["repetition_4gram"] for r in usable) / len(usable),
                "generation_seconds": generation_seconds,
                "seconds": time.perf_counter() - step_started,
                "rollout_hash": rollout_hash,
            }
            steps_log.append(record)
            print(
                f"step {step}: loss={record['loss']:.5f} lr={record['lr']:.2e} "
                f"grad={record['grad_norm']:.2f} rollouts={len(usable)} "
                f"tokens={total_tokens} gen={generation_seconds:.1f}s "
                f"total={record['seconds']:.1f}s"
            )
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate_heldout(
                    student, teacher, blocks, args.eval_batch_size, args.chunk_size
                )
                validation_curve.append({"step": step, "lr": record["lr"], **metrics})
                print(f"step {step} validation: {metrics}")

    elapsed = time.perf_counter() - started
    response_tokens_seen = sum(s["response_tokens"] for s in steps_log)
    checkpoint_path = ckpt_dir / f"{args.run_name}_step{args.steps}.pt"
    checkpoint_check: dict[str, object] = {"saved": False}
    if not aborted:
        meta = {
            "run_name": args.run_name,
            "model": args.model,
            "revisions": revisions,
            "quantization": quant_config.to_dict(),
            "data": {"prompts_sha256": prompt_hash, "eval_sha256": eval_hash},
            "steps": args.steps,
            "mode": "on_policy_forward_kl",
        }
        checkpoint_bytes = save_checkpoint(checkpoint_path, student, optimizer, args.steps, meta)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sample_key = next(name for name in payload["model"] if name.endswith("q_proj.weight"))
        checkpoint_check = {
            "saved": True,
            "path": str(checkpoint_path),
            "bytes": checkpoint_bytes,
            "keys_match": set(payload["model"]) == set(student.state_dict()),
            "sample_key": sample_key,
            "sample_max_abs_diff": float(
                (student.state_dict()[sample_key].cpu() - payload["model"][sample_key]).abs().max()
            ),
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
        "mode": "on_policy_forward_kl",
        "trainable_parameters": trainable_params,
        "wrapped_modules": len(wrapped),
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "scheduler": "linear warmup + cosine to 10%",
        },
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "prompts_per_step": args.prompts_per_step,
            "generation_batch_size": args.generation_batch_size,
            "seed_rule": "seed + step*1000 + batch_start",
        },
        "data": {
            "prompts": args.prompts,
            "prompts_sha256": prompt_hash,
            "eval": args.eval,
            "eval_sha256": eval_hash,
        },
        "train_config": {
            "steps": args.steps,
            "max_seq_len": args.max_seq_len,
            "warmup_steps": args.warmup_steps,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "budget": {
            "response_tokens_seen": response_tokens_seen,
            "q1_response_tokens": q1_tokens,
            "ratio_vs_q1": (response_tokens_seen / q1_tokens) if q1_tokens else None,
        },
        "validation_curve": validation_curve,
        "steps_log": steps_log,
        "rollout_hashes": rollout_hashes[:5] + rollout_hashes[-3:],
        "rollout_refreshed": len(set(rollout_hashes)) > 1,
        "elapsed_seconds": elapsed,
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
            "rollout_refreshed": len(set(rollout_hashes)) > 1,
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
        "warnings": (["short run: pilot budget"] if args.steps <= 250 else []),
    }
    (out_dir / f"{args.run_name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out_dir / f"{args.run_name}_steps.jsonl").open("w") as handle:
        for record in steps_log:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(payload["assertions"], indent=2))
    print(f"budget: {payload['budget']}")
    print(f"checkpoint: {checkpoint_check}")
    print(f"written {out_dir / (args.run_name + '.json')}")


if __name__ == "__main__":
    main()
