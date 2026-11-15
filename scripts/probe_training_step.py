"""Short offline-QAD gradient/memory probe (no full training).

Runs a few AdamW steps of teacher-student forward KL on fixed prompt+gold
completion pairs with the chunked hidden-state loss. Checks that teacher
gradients are absent, student gradients are finite, quantization stays active,
and master weights actually change. Saves loss/gradient/memory records.
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

import torch

from qwen_qad.losses import (
    chunked_forward_kl,
    gather_scored_tokens,
    next_token_score_mask,
)
from qwen_qad.memory import MemorySampler, parameter_bytes
from qwen_qad.models import (
    load_model,
    load_tokenizer,
    resolve_revisions,
    set_quantization_enabled,
    set_trainable_quantized_only,
    wrap_quantized_linears,
)
from qwen_qad.quantization import FakeQuantLinear, QuantConfig


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_prompts(path: Path, count: int) -> list[dict]:
    prompts = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            prompts.append(record)
            if len(prompts) >= count:
                break
    if not prompts:
        raise RuntimeError("no prompts loaded")
    return prompts


def build_batch(record: dict, max_seq_len: int, device: torch.device):
    prompt_ids = [int(t) for t in record["prompt_ids"]]
    response_ids = [int(t) for t in record["response_ids"]]
    prompt_len = len(prompt_ids)
    if prompt_len >= max_seq_len:
        return None
    response_len = min(len(response_ids), max_seq_len - prompt_len)
    if response_len < 2:
        return None
    ids = torch.tensor([prompt_ids + response_ids[:response_len]], dtype=torch.long, device=device)
    mask = next_token_score_mask(
        torch.tensor([prompt_len], device=device),
        torch.tensor([response_len], device=device),
        ids.shape[1],
        device=device,
    )
    return ids, mask, prompt_len, response_len


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--prompts", default="data/cache/prompts/openthoughts_prompts.jsonl")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--out", default="experiments/gate2/training_probe.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    revisions = resolve_revisions(args.model, args.revision)
    tokenizer = load_tokenizer(args.model, args.revision)

    teacher = load_model(args.model, args.revision, device=device, dtype=dtype)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    quant_config = QuantConfig(bits=args.bits, group_size=args.group_size)
    student = load_model(args.model, args.revision, device=device, dtype=dtype)
    replaced = wrap_quantized_linears(student, quant_config)
    trainable_params = set_trainable_quantized_only(student)
    set_quantization_enabled(student, True)
    student.train()

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), foreach=False)
    teacher_lm = teacher.get_output_embeddings().weight.detach()
    student_lm = student.get_output_embeddings().weight.detach()

    prompts_all = load_prompts(Path(args.prompts), max(args.steps * 4, 16))
    prompts = [
        p
        for p in prompts_all
        if len(p["prompt_ids"]) + 32 <= args.max_seq_len
    ]
    if not prompts:
        raise RuntimeError(
            f"no prompt fits max_seq_len={args.max_seq_len}; "
            f"available prompt lengths: {sorted(len(p['prompt_ids']) for p in prompts_all)[:10]}"
        )
    prompt_hash = hashlib.sha256(
        json.dumps([p["prompt_ids"] for p in prompts], sort_keys=True).encode()
    ).hexdigest()

    reference_weight = next(m for m in student.modules() if isinstance(m, FakeQuantLinear))
    before = reference_weight.weight.detach().clone()

    steps = []
    started = time.perf_counter()
    with MemorySampler(device) as sampler:
        for step in range(args.steps):
            batch = build_batch(prompts[step % len(prompts)], args.max_seq_len, device)
            if batch is None:
                raise RuntimeError("prompt too long for max-seq-len")
            ids, mask, prompt_len, response_len = batch
            step_started = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_hidden = teacher.model(input_ids=ids, use_cache=False).last_hidden_state
                h_teacher, targets = gather_scored_tokens(teacher_hidden, ids, mask)
            student_hidden = student.model(input_ids=ids, use_cache=False).last_hidden_state
            h_student, _ = gather_scored_tokens(student_hidden, ids, mask)
            loss = chunked_forward_kl(
                h_student, h_teacher, student_lm, teacher_lm, chunk_size=args.chunk_size
            )
            loss.backward()

            grad_norm = torch.sqrt(
                sum(
                    (p.grad.detach().float() ** 2).sum()
                    for p in trainable
                    if p.grad is not None
                )
            )
            grad_finite = bool(torch.isfinite(grad_norm))
            optimizer.step()
            for module in student.modules():
                if isinstance(module, FakeQuantLinear):
                    module.invalidate_cache()

            steps.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "grad_norm": float(grad_norm),
                    "grad_finite": grad_finite,
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "tokens_scored": int(targets.numel()),
                    "step_seconds": time.perf_counter() - step_started,
                }
            )
            print(
                f"step {step}: loss={float(loss.detach()):.6f} grad_norm={float(grad_norm):.3e} "
                f"responses={response_len} {steps[-1]['step_seconds']:.1f}s"
            )

    elapsed = time.perf_counter() - started
    after = reference_weight.weight.detach()
    teacher_grads = [p.grad for p in teacher.parameters() if p.grad is not None]
    student_grads = [p.grad for p in trainable if p.grad is not None]
    quantized_active = sum(
        1 for m in student.modules() if isinstance(m, FakeQuantLinear) and m.quantized
    )
    changed = float((after - before).abs().max()) > 0.0

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": {"id": args.model, "revisions": revisions},
        "quantization": quant_config.to_dict(),
        "trainable_parameters": trainable_params,
        "wrapped_modules": len(replaced),
        "optimizer": {"name": "AdamW", "lr": args.lr, "betas": [0.9, 0.95], "weight_decay": 0.0},
        "prompts": {"path": args.prompts, "ids_sha256": prompt_hash, "steps": args.steps, "usable": len(prompts)},
        "max_seq_len": args.max_seq_len,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "tokens_per_second": sum(s["tokens_scored"] for s in steps) / elapsed,
        "memory": sampler.result(),
        "parameter_bytes": {"teacher": parameter_bytes(teacher), "student": parameter_bytes(student)},
        "assertions": {
            "teacher_gradients_absent": len(teacher_grads) == 0,
            "student_gradients_finite": all(
                g is None or bool(torch.isfinite(g).all()) for g in student_grads
            ),
            "quantized_forward_active": quantized_active == len(replaced),
            "master_weights_changed": changed,
            "loss_finite": all(math.isfinite(s["loss"]) for s in steps),
        },
        "environment": {
            "device": str(device),
            "dtype": args.dtype,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "notes": [
            "Probe only: a few steps on fixed gold completions, not QAD training.",
            "Memory is process-level; parameter bytes are reported separately.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["assertions"], indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
