"""Compact GSM8K accuracy check for T0/Q0/Q1/Q2 (100 test examples).

Descriptive pilot metric: thinking-mode generation capped at 512 tokens, final
numeric answer extracted from ``####`` (or the last number). Parse failures are
reported separately and never counted as correct.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

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

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def normalize_answer(text: str) -> str | None:
    text = text.strip().replace(",", "").replace("$", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    value = match.group(0)
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    except ValueError:
        return None


def extract_prediction(text: str) -> str | None:
    if "####" in text:
        return normalize_answer(text.split("####")[-1])
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace(",", ""))
    return normalize_answer(numbers[-1]) if numbers else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--bits", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    revisions = resolve_revisions(args.model, args.revision)
    tokenizer = load_tokenizer(args.model, args.revision)

    dataset = load_dataset("openai/gsm8k", "main", split="test", revision=GSM8K_REVISION)
    examples = list(dataset.select(range(args.count)))
    questions = [example["question"] for example in examples]
    golds = [normalize_answer(example["answer"].split("####")[-1]) for example in examples]

    prompt_ids_list = []
    for question in questions:
        messages = [
            {
                "role": "user",
                "content": (
                    question
                    + "\n\nSolve the problem step by step. Put the final numeric "
                    "answer after '####'."
                ),
            }
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
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
            max_new_tokens=args.max_new_tokens,
            batch_size=args.generation_batch_size,
            seed=1234,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    elapsed = time.perf_counter() - started

    records = []
    correct = 0
    parsed = 0
    for example, gold, rollout in zip(examples, golds, rollouts):
        text = tokenizer.decode(rollout["response_ids"], skip_special_tokens=True)
        prediction = extract_prediction(text)
        parsed += int(prediction is not None)
        is_correct = prediction is not None and gold is not None and prediction == gold
        correct += int(is_correct)
        records.append(
            {
                "question": example["question"][:200],
                "gold": gold,
                "prediction": prediction,
                "correct": is_correct,
                "response_len": rollout["response_len"],
                "text": text[:600],
            }
        )

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config_name": args.config_name,
        "model": {"id": args.model, "revisions": revisions},
        "checkpoint": args.checkpoint,
        "quantization": quant_config.to_dict() if quant_config else None,
        "dataset": {
            "name": "openai/gsm8k",
            "config": "main",
            "split": "test",
            "revision": GSM8K_REVISION,
            "count": len(examples),
            "license": "mit",
        },
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "thinking": True,
            "seed": 1234,
        },
        "metrics": {
            "accuracy": correct / len(examples),
            "correct": correct,
            "parsed": parsed,
            "parse_rate": parsed / len(examples),
            "mean_response_len": sum(r["response_len"] for r in records) / len(records),
            "elapsed_seconds": elapsed,
        },
        "memory": sampler.result(),
        "parameter_bytes": parameter_bytes(model),
        "records": records,
        "environment": {
            "device": str(device),
            "dtype": args.dtype,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "notes": [
            "Pilot subset: 100 GSM8K test examples, thinking mode, 512-token cap.",
            "Parse failures are reported and count as incorrect.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
