"""Gate 1.5 diagnostics: is the W3A16 degradation real, and where does it come from?

Runs on the pinned Qwen3-0.6B checkpoint with a fixed, small held-out subset:

1. wrapped model with quantization disabled vs the original BF16 logits/NLL;
2. BF16, W4A16-G128, W3A16-G32/G64/G128;
3. W3A16-G128 attention-only vs MLP-only;
4. NLL, perplexity, teacher KL, coverage, weight and activation diagnostics;
5. teacher self-KL computed from two independent forward passes;
6. resolved model/tokenizer revisions;
7. process memory reported as current/sampled-peak, with parameter bytes
   separately (teacher+student process is not comparable to a single-model run).

This is a diagnostic subset, not the official Gate 2 evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qwen_qad.diagnostics import (
    capture_hidden_states,
    compare_activations,
    quantization_coverage,
    weight_diagnostics,
)
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
    set_quantization_enabled,
    set_trainable_quantized_only,
    wrap_quantized_linears,
)
from qwen_qad.quantization import QuantConfig

CONFIGS = [
    {"name": "bf16", "kind": "reference"},
    {"name": "w4a16_g128", "bits": 4, "group_size": 128},
    {"name": "w3a16_g32", "bits": 3, "group_size": 32},
    {"name": "w3a16_g64", "bits": 3, "group_size": 64},
    {"name": "w3a16_g128", "bits": 3, "group_size": 128},
    {
        "name": "w3a16_g128_attn",
        "bits": 3,
        "group_size": 128,
        "include": r".*self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    },
    {
        "name": "w3a16_g128_mlp",
        "bits": 3,
        "group_size": 128,
        "include": r".*mlp\.(gate_proj|up_proj|down_proj)",
    },
]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def iter_batches(blocks: np.ndarray, batch_size: int, device: torch.device):
    for start in range(0, blocks.shape[0], batch_size):
        yield torch.tensor(blocks[start : start + batch_size], dtype=torch.long, device=device)


def score_mask(input_ids: torch.Tensor) -> torch.Tensor:
    prompt_lens = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    response_lens = torch.full_like(prompt_lens, input_ids.shape[1] - 1)
    return next_token_score_mask(
        prompt_lens, response_lens, input_ids.shape[1], device=input_ids.device
    )


def tensor_bytes(items) -> int:
    total = 0
    for item in items:
        if isinstance(item, torch.Tensor):
            total += item.numel() * item.element_size()
        elif isinstance(item, dict):
            total += tensor_bytes(item.values())
        elif isinstance(item, (list, tuple)):
            total += tensor_bytes(item)
    return total


def build_teacher_cache(model, blocks, batch_size, device, layers):
    cache = []
    for input_ids in iter_batches(blocks, batch_size, device):
        hidden, layer_map = capture_hidden_states(model, input_ids, layers)
        h_scored, targets = gather_scored_tokens(hidden, input_ids, score_mask(input_ids))
        cache.append(
            {"h_scored": h_scored, "targets": targets, "layers": layer_map, "batch": input_ids.shape[0]}
        )
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--eval", default="data/cache/eval/fineweb_eval_ids.npy")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit-batches", type=int, default=16)
    parser.add_argument("--layers", default="0,13,27")
    parser.add_argument("--equivalence-sequences", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--out-dir", default="experiments/gate15")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    layers = [int(x) for x in args.layers.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    revisions = resolve_revisions(args.model, args.revision)
    tokenizer = load_tokenizer(args.model, args.revision)
    teacher = load_model(args.model, args.revision, device=device, dtype=dtype)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_lm = teacher.get_output_embeddings().weight.detach()

    blocks = np.load(args.eval)
    total_blocks = blocks.shape[0]
    if args.limit_batches is not None:
        blocks = blocks[: args.batch_size * args.limit_batches]

    started = time.perf_counter()
    teacher_cache = build_teacher_cache(teacher, blocks, args.batch_size, device, layers)
    cache_bytes = tensor_bytes(teacher_cache)

    results: dict[str, dict] = {}
    reference: dict | None = None

    for spec in CONFIGS:
        name = spec["name"]
        config_started = time.perf_counter()
        is_reference = spec.get("kind") == "reference"

        if is_reference:
            student = teacher
            student_lm = teacher_lm
            quant_config = None
            memory = None
            coverage = None
            weight_metrics = None
        else:
            quant_config = QuantConfig(bits=spec["bits"], group_size=spec["group_size"])
            student = load_model(args.model, args.revision, device=device, dtype=dtype)
            include = [spec["include"]] if "include" in spec else None
            replaced = wrap_quantized_linears(student, quant_config, include=include)
            set_trainable_quantized_only(student)
            set_quantization_enabled(student, True)
            student_lm = student.get_output_embeddings().weight.detach()
            coverage = quantization_coverage(student, quant_config)
            coverage["wrapped_modules"] = len(replaced)
            weight_metrics = weight_diagnostics(student, quant_config)
            memory = None
            if device.type == "mps":
                torch.mps.empty_cache()

        nll_sum = 0.0
        kl_sum = 0.0
        tokens = 0
        layer_errors: dict[int, list[float]] = {index: [] for index in layers}
        layer_max_errors: dict[int, float] = {index: 0.0 for index in layers}
        layer_cosines: dict[int, list[float]] = {index: [] for index in layers}
        self_kl_sum = 0.0
        self_kl_max = 0.0
        second_pass_seconds = 0.0

        sampler = MemorySampler(device)
        with sampler:
            for batch, cached in zip(iter_batches(blocks, args.batch_size, device), teacher_cache):
                if is_reference:
                    h_student = cached["h_scored"]
                    teacher_layers = cached["layers"]
                    # Second independent forward pass for a numerically computed self-KL.
                    pass_started = time.perf_counter()
                    hidden2, layers2 = capture_hidden_states(teacher, batch, layers)
                    second_pass_seconds += time.perf_counter() - pass_started
                    h2, targets = gather_scored_tokens(hidden2, batch, score_mask(batch))
                    batch_self_kl = chunked_forward_kl(
                        h2, cached["h_scored"], teacher_lm, teacher_lm, chunk_size=args.chunk_size
                    )
                    self_kl_sum += float(batch_self_kl) * targets.numel()
                    self_kl_max = max(self_kl_max, abs(float(batch_self_kl)))
                    student_layers = layers2
                else:
                    hidden_s, student_layers = capture_hidden_states(student, batch, layers)
                    h_student, targets = gather_scored_tokens(
                        hidden_s, batch, score_mask(batch)
                    )
                    teacher_layers = cached["layers"]
                    batch_self_kl = None

                nll = chunked_nll(h_student, cached["targets"], student_lm, chunk_size=args.chunk_size)
                if is_reference:
                    kl = nll.new_zeros(())
                else:
                    kl = chunked_forward_kl(
                        h_student,
                        cached["h_scored"],
                        student_lm,
                        teacher_lm,
                        chunk_size=args.chunk_size,
                    )
                count = int(cached["targets"].numel())
                nll_sum += float(nll) * count
                kl_sum += float(kl) * count
                tokens += count

                comparisons = compare_activations(student_layers, teacher_layers)
                for index in layers:
                    layer_errors[index].append(comparisons[str(index)]["relative_error"])
                    layer_cosines[index].append(comparisons[str(index)]["cosine_similarity"])
                    layer_max_errors[index] = max(
                        layer_max_errors[index], comparisons[str(index)]["relative_error"]
                    )

        elapsed = time.perf_counter() - config_started
        nll_mean = nll_sum / tokens
        result = {
            "config": spec,
            "quantized": not is_reference,
            "nll": nll_mean,
            "perplexity": math.exp(nll_mean),
            "forward_kl_teacher_student": kl_sum / tokens,
            "teacher_self_kl": (self_kl_sum / tokens) if is_reference else None,
            "teacher_self_kl_max_batch": self_kl_max if is_reference else None,
            "tokens": tokens,
            "activation_relative_error_mean": {
                str(index): float(np.mean(values)) for index, values in layer_errors.items()
            },
            "activation_relative_error_max": {
                str(index): float(value) for index, value in layer_max_errors.items()
            },
            "activation_cosine_mean": {
                str(index): float(np.mean(values)) for index, values in layer_cosines.items()
            },
            "activation_cosine_min": {
                str(index): float(np.min(values)) for index, values in layer_cosines.items()
            },
            "coverage": coverage,
            "weight_diagnostics": weight_metrics,
            "parameter_bytes": {
                "teacher": parameter_bytes(teacher),
                "student": parameter_bytes(student),
                "teacher_cache": cache_bytes,
            },
            "memory": sampler.result(),
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens / elapsed,
        }
        if reference is None:
            reference = result
        else:
            result["nll_delta_vs_bf16"] = result["nll"] - reference["nll"]
            result["perplexity_ratio_vs_bf16"] = result["perplexity"] / reference["perplexity"]
        results[name] = result
        print(
            f"{name:20s} nll={result['nll']:.4f} ppl={result['perplexity']:.2f} "
            f"kl={result['forward_kl_teacher_student']:.4f} "
            f"actL0={result['activation_relative_error_mean'].get('0', 0):.3e} "
            f"time={elapsed:.0f}s"
        )

        if not is_reference:
            del student
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()

    # ---- quantization-disabled equivalence on the fully wrapped model -----
    equivalence_student = load_model(args.model, args.revision, device=device, dtype=dtype)
    full_config = QuantConfig(bits=3, group_size=128)
    replaced = wrap_quantized_linears(equivalence_student, full_config)
    set_quantization_enabled(equivalence_student, False)
    n_equiv = min(args.equivalence_sequences, blocks.shape[0])
    max_logit_diff = 0.0
    mean_logit_diff = 0.0
    nll_teacher = 0.0
    nll_wrapped = 0.0
    equivalence_tokens = 0
    with MemorySampler(device) as equivalence_memory:
        for index in range(n_equiv):
            ids = torch.tensor(blocks[index : index + 1], dtype=torch.long, device=device)
            with torch.no_grad():
                teacher_logits = teacher(ids).logits.float()
                wrapped_logits = equivalence_student(ids).logits.float()
            diff = (teacher_logits - wrapped_logits).abs()
            max_logit_diff = max(max_logit_diff, float(diff.max()))
            mean_logit_diff += float(diff.mean())
            nll_teacher += float(
                F.cross_entropy(
                    teacher_logits[:, :-1].reshape(-1, teacher_logits.shape[-1]),
                    ids[:, 1:].reshape(-1),
                )
            )
            nll_wrapped += float(
                F.cross_entropy(
                    wrapped_logits[:, :-1].reshape(-1, wrapped_logits.shape[-1]),
                    ids[:, 1:].reshape(-1),
                )
            )
            equivalence_tokens += ids[:, 1:].numel()
    equivalence = {
        "wrapped_modules": len(replaced),
        "wrapped_modules_quantized_flag": sum(
            1 for m in equivalence_student.modules() if hasattr(m, "quantized") and m.quantized
        ),
        "sequences": n_equiv,
        "max_abs_logit_diff": max_logit_diff,
        "mean_abs_logit_diff": mean_logit_diff / n_equiv,
        "nll_teacher": nll_teacher / n_equiv,
        "nll_wrapped_disabled": nll_wrapped / n_equiv,
        "nll_abs_diff": abs(nll_teacher - nll_wrapped) / n_equiv,
        "tokens": equivalence_tokens,
        "memory": equivalence_memory.result(),
        "tolerance": "max_abs_logit_diff == 0.0 expected (fp32 master is a lossless upcast)",
    }
    del equivalence_student
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": {"id": args.model, "revisions": revisions},
        "tokenizer": {
            "id": args.model,
            "vocab_size": int(tokenizer.vocab_size),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "environment": {
            "device": str(device),
            "dtype": args.dtype,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "platform": platform.platform(),
        },
        "eval_set": {
            "path": args.eval,
            "total_blocks_available": int(total_blocks),
            "blocks_used": int(blocks.shape[0]),
            "block_size": int(blocks.shape[1]),
            "tokens_scored_per_config": int(sum(r["tokens"] for r in results.values()) // len(results)),
            "batch_size": args.batch_size,
            "layers_probed": layers,
        },
        "configs": results,
        "quantization_disabled_equivalence": equivalence,
        "notes": [
            "Diagnostic subset only; not the official Gate 2 evaluation.",
            "MPS peak memory is sampled at 20 ms and is a lower bound; CUDA would use max_memory_allocated.",
            "Memory is process-level and includes teacher+student; compare parameter_bytes for model sizes.",
            "Teacher self-KL is computed between two independent forward passes, not assigned.",
            "W4A16 uses the same symmetric per-group quantizer with b=4 and G=128.",
        ],
    }
    json_path = out_dir / "gate15_diagnostics.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Gate 1.5 diagnostics",
        "",
        f"- model `{args.model}` @ `{revisions.get('resolved_model_sha')}`",
        f"- device `{device}`, dtype `{args.dtype}`, eval blocks {blocks.shape[0]} × {blocks.shape[1]}",
        f"- git `{git_commit()[:12]}`, created {payload['created_utc']}",
        "",
        "| config | NLL | ppl | KL(T||S) | quantized params | eff bits | weight MSE | act L0 | act L13 | act L27 | cos L0 | cos L27 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        coverage = result["coverage"]
        lines.append(
            "| {name} | {nll:.4f} | {ppl:.2f} | {kl:.4f} | {q:.1%} | {bits} | {mse} | {a0} | {a13} | {a27} | {c0} | {c27} |".format(
                name=name,
                nll=result["nll"],
                ppl=result["perplexity"],
                kl=result["forward_kl_teacher_student"],
                q=(coverage["fraction_of_model_parameters"] if coverage else 0.0),
                bits=(f"{coverage['effective_bits_per_quantized_weight']:.3f}" if coverage else "—"),
                mse=(f"{result['weight_diagnostics']['mse_weighted']:.2e}" if result["weight_diagnostics"] else "—"),
                a0=f"{result['activation_relative_error_mean'].get('0', 0):.2e}",
                a13=f"{result['activation_relative_error_mean'].get('13', 0):.2e}",
                a27=f"{result['activation_relative_error_mean'].get('27', 0):.2e}",
                c0=f"{result['activation_cosine_mean'].get('0', 0):.4f}",
                c27=f"{result['activation_cosine_mean'].get('27', 0):.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## Quantization-disabled equivalence",
            "",
            f"- wrapped modules: {equivalence['wrapped_modules']}",
            f"- sequences: {equivalence['sequences']}",
            f"- max abs logit diff: {equivalence['max_abs_logit_diff']:.3e}",
            f"- mean abs logit diff: {equivalence['mean_abs_logit_diff']:.3e}",
            f"- NLL teacher: {equivalence['nll_teacher']:.6f}",
            f"- NLL wrapped/disabled: {equivalence['nll_wrapped_disabled']:.6f}",
            f"- NLL abs diff: {equivalence['nll_abs_diff']:.3e}",
            "",
            "## Teacher self-KL (two independent passes)",
            "",
            f"- mean: {reference['teacher_self_kl']:.3e}",
            f"- max batch: {reference['teacher_self_kl_max_batch']:.3e}",
            "",
            "Full machine-readable results: `gate15_diagnostics.json`.",
        ]
    )
    md_path = out_dir / "diagnostics.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nwritten {json_path} and {md_path}")
    print(f"total elapsed {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
