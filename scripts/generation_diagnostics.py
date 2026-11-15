"""Fixed-prompt generation diagnostics for BF16 and quantized students.

Identical prompts, sampling settings and seed across configurations. Reports
repetition, premature EOS, hit-cap rate, empty outputs and saved examples.
Descriptive only: sampling is stochastic and 8 prompts are not a benchmark.
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

from qwen_qad.generation import generation_summary
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

CONFIG_SPECS = {
    "bf16": None,
    "w3a16_g128": {"bits": 3, "group_size": 128},
    "w3a16_g32": {"bits": 3, "group_size": 32},
    "w4a16_g128": {"bits": 4, "group_size": 128},
}


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
            prompts.append({"prompt_ids": record["prompt_ids"], "question": record["question"]})
            if len(prompts) >= count:
                break
    if len(prompts) < count:
        raise RuntimeError(f"only {len(prompts)} prompts available, wanted {count}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--prompts", default="data/cache/prompts/openthoughts_prompts.jsonl")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", default="bf16,w3a16_g128,w3a16_g32,w4a16_g128")
    parser.add_argument("--out", default="experiments/gate2/generation_diagnostics.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    config_names = [name.strip() for name in args.configs.split(",") if name.strip()]
    unknown = [name for name in config_names if name not in CONFIG_SPECS]
    if unknown:
        raise ValueError(f"unknown configs: {unknown}; choose {sorted(CONFIG_SPECS)}")

    revisions = resolve_revisions(args.model, args.revision)
    tokenizer = load_tokenizer(args.model, args.revision)
    prompts = load_prompts(Path(args.prompts), args.num_prompts)
    prompt_hash = hashlib.sha256(
        json.dumps([p["prompt_ids"] for p in prompts], sort_keys=True).encode()
    ).hexdigest()

    results: dict[str, dict] = {}
    for name in config_names:
        spec = CONFIG_SPECS[name]
        model = load_model(args.model, args.revision, device=device, dtype=dtype)
        quant_config = None
        if spec is not None:
            quant_config = QuantConfig(bits=spec["bits"], group_size=spec["group_size"])
            wrap_quantized_linears(model, quant_config)
            set_trainable_quantized_only(model)
            set_quantization_enabled(model, True)

        records = []
        started = time.perf_counter()
        with MemorySampler(device) as sampler:
            for index, prompt in enumerate(prompts):
                ids = torch.tensor([prompt["prompt_ids"]], dtype=torch.long, device=device)
                torch.manual_seed(args.seed + index)
                with torch.no_grad():
                    generated = model.generate(
                        ids,
                        attention_mask=torch.ones_like(ids),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                token_ids = generated[0].tolist()
                summary = generation_summary(
                    token_ids,
                    prompt_len=len(prompt["prompt_ids"]),
                    eos_token_id=tokenizer.eos_token_id,
                    min_new_tokens=args.min_new_tokens,
                )
                continuation = token_ids[len(prompt["prompt_ids"]) :]
                records.append(
                    {
                        "prompt_index": index,
                        "question": prompt["question"][:300],
                        "summary": summary,
                        "text": tokenizer.decode(continuation, skip_special_tokens=False)[:1000],
                    }
                )
        elapsed = time.perf_counter() - started

        def mean(key: str) -> float:
            return sum(float(r["summary"][key]) for r in records) / len(records)

        results[name] = {
            "quantization": quant_config.to_dict() if quant_config else None,
            "parameter_bytes": parameter_bytes(model),
            "records": records,
            "aggregate": {
                "mean_generated_tokens": mean("generated_tokens"),
                "mean_content_tokens": mean("content_tokens"),
                "mean_repetition_4gram": mean("repetition_4gram"),
                "mean_max_repeated_4gram": mean("max_repeated_4gram"),
                "mean_distinct_2": mean("distinct_2"),
                "premature_eos_count": sum(r["summary"]["premature_eos"] for r in records),
                "hit_max_tokens_count": sum(r["summary"]["hit_max_tokens"] for r in records),
                "empty_count": sum(r["summary"]["empty"] for r in records),
                "eos_rate": sum(r["summary"]["eos_generated"] for r in records) / len(records),
            },
            "elapsed_seconds": elapsed,
            "memory": sampler.result(),
        }
        print(
            f"{name:16s} len={results[name]['aggregate']['mean_generated_tokens']:.0f} "
            f"rep4={results[name]['aggregate']['mean_repetition_4gram']:.3f} "
            f"premature={results[name]['aggregate']['premature_eos_count']} "
            f"cap={results[name]['aggregate']['hit_max_tokens_count']} "
            f"time={elapsed:.0f}s"
        )
        del model
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
        "prompts": {
            "path": args.prompts,
            "num_prompts": len(prompts),
            "ids_sha256": prompt_hash,
        },
        "generation": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "seed_per_prompt": f"{args.seed}+prompt_index (identical across configs)",
            "note": "fixed seed does not guarantee identical trajectories across different models",
        },
        "environment": {
            "device": str(device),
            "dtype": args.dtype,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "configs": results,
        "notes": [
            "Descriptive only: 8 prompts, one seed, stochastic sampling.",
            "Prompts are the first N rows of the pinned OpenThoughts pool; identical for all configs.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Generation-quality diagnostics",
        "",
        f"- model `{args.model}` @ `{revisions.get('resolved_model_sha')}`",
        f"- {len(prompts)} fixed prompts, T=0.6, top_p=0.95, top_k=20, "
        f"max_new_tokens={args.max_new_tokens}, seed={args.seed}",
        f"- git `{git_commit()[:12]}`",
        "",
        "| config | mean tokens | repetition(4g) | premature EOS | hit cap | empty | EOS rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        agg = result["aggregate"]
        lines.append(
            f"| {name} | {agg['mean_generated_tokens']:.0f} | {agg['mean_repetition_4gram']:.3f} | "
            f"{agg['premature_eos_count']}/{len(prompts)} | {agg['hit_max_tokens_count']}/{len(prompts)} | "
            f"{agg['empty_count']}/{len(prompts)} | {agg['eos_rate']:.1%} |"
        )
    lines.extend(["", "## Examples", ""])
    for name, result in results.items():
        lines.append(f"### {name}")
        for record in result["records"][:2]:
            lines.append(f"- prompt: {record['question'][:160]!r}")
            lines.append(f"  - {record['text'][:500]!r}")
        lines.append("")
    (out.parent / "generation_diagnostics.md").write_text("\n".join(lines) + "\n")
    print(f"written {out}")


if __name__ == "__main__":
    main()
