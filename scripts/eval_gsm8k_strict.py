"""Strict GSM8K evaluation (non-thinking and thinking modes).

Only explicit final answers are graded. Thinking-mode generations must close
``</think>`` before an answer region exists; a cap-hit generation is reported as
truncated and never graded from intermediate numbers. Every example is saved
with prompt/response token ids and full decoded text for independent regrading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from qwen_qad.answer_parsing import normalize_number, parse_final_answer
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
from qwen_qad.rollouts import generate_rollouts
from qwen_qad.stats import bootstrap_ci

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
INSTRUCTION = (
    "\n\nSolve the problem step by step. Put the final numeric answer on its own "
    "line exactly as: #### <number>"
)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def prompt_for(tokenizer, question: str, thinking: bool) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question + INSTRUCTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--mode", choices=["nonthinking", "thinking"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--bits", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    thinking = args.mode == "thinking"
    max_new = args.max_new_tokens or (4096 if thinking else 256)
    batch_size = args.generation_batch_size or (8 if thinking else 32)

    from datasets import load_dataset

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    revisions = resolve_revisions(args.model, args.revision)
    tokenizer = load_tokenizer(args.model, args.revision)

    dataset = load_dataset("openai/gsm8k", "main", split="test", revision=GSM8K_REVISION)
    examples = list(dataset.select(range(args.offset, args.offset + args.count)))
    golds = [normalize_number(example["answer"].split("####")[-1]) for example in examples]
    prompt_ids_list = []
    for example in examples:
        text = prompt_for(tokenizer, example["question"], thinking)
        prompt_ids_list.append(tokenizer(text, add_special_tokens=False).input_ids)

    model = load_model(args.model, args.revision, device=device, dtype=dtype)
    quant_config = None
    if args.checkpoint or args.bits:
        bits = args.bits if args.bits is not None else 4
        quant_config = QuantConfig(bits=bits, group_size=args.group_size)
        wrap_quantized_linears(model, quant_config)
        set_trainable_quantized_only(model)
        set_quantization_enabled(model, True)
        if args.checkpoint:
            from qwen_qad.checkpoint import load_checkpoint

            load_checkpoint(args.checkpoint, model, None, map_location="cpu")

    started = time.perf_counter()
    with MemorySampler(device) as sampler:
        rollouts = generate_rollouts(
            model,
            tokenizer,
            prompt_ids_list,
            max_new_tokens=max_new,
            batch_size=batch_size,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elapsed = time.perf_counter() - started

    records = []
    correct_flags = []
    status_counts: dict[str, int] = {}
    completed = 0
    truncated = 0
    thinking_closed_count = 0
    for index, (example, gold, prompt_ids, rollout) in enumerate(
        zip(examples, golds, prompt_ids_list, rollouts)
    ):
        response_ids = rollout["response_ids"]
        finished = bool(response_ids) and response_ids[-1] == tokenizer.eos_token_id
        hit_cap = not finished and len(response_ids) >= max_new
        full_text = tokenizer.decode(response_ids, skip_special_tokens=False)
        parse_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        parsed = parse_final_answer(parse_text, thinking_mode=thinking)
        status_counts[parsed.status] = status_counts.get(parsed.status, 0) + 1
        completed += int(finished)
        truncated += int(hit_cap)
        thinking_closed_count += int(parsed.thinking_closed)
        is_correct = parsed.status == "ok" and parsed.value is not None and parsed.value == gold
        correct_flags.append(is_correct)
        records.append(
            {
                "example_id": f"gsm8k-test-{args.offset + index}",
                "question_sha256": hashlib.sha256(example["question"].encode()).hexdigest(),
                "question": example["question"],
                "gold_answer": gold,
                "gold_raw": example["answer"].split("####")[-1].strip(),
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "full_text": full_text,
                "parse_text": parse_text,
                "finished_naturally": finished,
                "hit_cap": hit_cap,
                "thinking_closed": parsed.thinking_closed,
                "parse_status": parsed.status,
                "parsed_value": parsed.value,
                "parse_marker": parsed.marker,
                "answer_region": parsed.answer_region,
                "correct": is_correct,
            }
        )

    point, low, high = bootstrap_ci(correct_flags, n_boot=1000, seed=args.seed)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_name": args.config_name,
        "mode": args.mode,
        "model": {"id": args.model, "revisions": revisions},
        "checkpoint": args.checkpoint,
        "quantization": quant_config.to_dict() if quant_config else None,
        "dataset": {
            "name": "openai/gsm8k",
            "config": "main",
            "split": "test",
            "revision": GSM8K_REVISION,
            "offset": args.offset,
            "count": len(examples),
            "license": "mit",
        },
        "generation": {
            "enable_thinking": thinking,
            "max_new_tokens": max_new,
            "batch_size": batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "instruction": INSTRUCTION.strip(),
        },
        "metrics": {
            "accuracy": point,
            "accuracy_ci95": [low, high],
            "correct": sum(correct_flags),
            "count": len(examples),
            "explicit_answer_rate": (len(examples) - status_counts.get("missing", 0)
                                     - status_counts.get("incomplete_thinking", 0)) / len(examples),
            "completion_rate": completed / len(examples),
            "truncation_rate": truncated / len(examples),
            "thinking_closed_rate": thinking_closed_count / len(examples),
            "parse_status_counts": status_counts,
            "mean_response_tokens": sum(len(r["response_ids"]) for r in records) / len(records),
            "elapsed_seconds": elapsed,
        },
        "memory": sampler.result(),
        "parameter_bytes": parameter_bytes(model),
        "environment": {
            "device": str(device),
            "dtype": args.dtype,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "notes": [
            "Strict grading: only explicit ####, \\boxed{}, or Answer: markers count.",
            "Thinking mode requires a closed </think>; unfinished reasoning is never graded.",
            "Full prompt/response token ids and decoded text are saved for regrading.",
        ],
    }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    with out_prefix.with_suffix(".generations.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(summary["metrics"], indent=2))
    print(f"written {out_prefix}.json and {out_prefix}.generations.jsonl")


if __name__ == "__main__":
    main()
