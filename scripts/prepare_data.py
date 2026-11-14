"""Gate 1 data preparation: fixed held-out text and a fixed prompt pool.

Writes:
  data/cache/eval/fineweb_eval_ids.npy        [n_blocks, block_size] int32
  data/cache/prompts/openthoughts_prompts.jsonl
  data/manifests/gate1_data_manifest.json

Everything is pinned by dataset revision and content hash. FineWeb-Edu is used
only for teacher-forced NLL/KL, never for training. OpenThoughts prompts are
used for Q1 (with the truncated gold completion) and Q2/Q3 (student rollouts).

Gold OpenThoughts completions are much longer than the pilot budget (median
~7.3k tokens), so responses are truncated to ``--max-response-tokens`` and the
truncation is recorded per example and counted in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"

OT_DATASET = "open-thoughts/OpenThoughts-114k"
OT_REVISION = "bd093c3994fd54d2390985b66988ddf282a55eb6"
OT_PARQUET = "data/train-00000-of-00006.parquet"

TOKENIZER_ID = "Qwen/Qwen3-0.6B"
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: object) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True).encode("utf-8"))


def fetch_viewer_rows(
    dataset: str, config: str, split: str, offsets: list[int], length: int
) -> list[dict]:
    rows: list[dict] = []
    for offset in offsets:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = json.load(response)
        rows.extend(item["row"] for item in payload["rows"])
    return rows


def build_fineweb_eval(
    tokenizer, n_blocks: int, block_size: int, docs: int
) -> tuple[np.ndarray, dict]:
    offsets = list(range(0, docs, 100))
    rows = fetch_viewer_rows(FINEWEB_DATASET, FINEWEB_CONFIG, "train", offsets, 100)
    texts = [row["text"] for row in rows]
    doc_ids = [row.get("id") for row in rows]

    eos = tokenizer.eos_token_id
    flat: list[int] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        flat.extend(ids)
        flat.append(eos)
    total_tokens = len(flat)
    if total_tokens < n_blocks * block_size:
        raise RuntimeError(
            f"not enough held-out tokens: {total_tokens} < {n_blocks * block_size}"
        )
    usable = (total_tokens // block_size) * block_size
    array = np.asarray(flat[:usable], dtype=np.int32).reshape(-1, block_size)
    array = array[:n_blocks]

    manifest = {
        "dataset": FINEWEB_DATASET,
        "config": FINEWEB_CONFIG,
        "revision": FINEWEB_REVISION,
        "split": "train",
        "license": "odc-by",
        "viewer_offsets": offsets,
        "docs_requested": docs,
        "docs_received": len(rows),
        "doc_ids_sha256": sha256_json(doc_ids),
        "block_size": block_size,
        "n_blocks": int(array.shape[0]),
        "num_tokens": int(array.size),
        "ids_sha256": sha256_bytes(array.tobytes()),
        "separator_token_id": eos,
    }
    return array, manifest


def build_prompt_pool(
    tokenizer,
    parquet_path: Path,
    n_prompts: int,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    truncated = 0
    skipped_prompt = 0
    row_index = -1

    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=256):
        for example in batch.to_pylist():
            row_index += 1
            user = next(
                (t["value"] for t in example["conversations"] if t["from"] == "user"), None
            )
            assistant = next(
                (t["value"] for t in example["conversations"] if t["from"] == "assistant"),
                None,
            )
            if user is None or assistant is None:
                continue

            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": example["system"]},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            if len(prompt_ids) > max_prompt_tokens:
                skipped_prompt += 1
                continue

            response_ids = tokenizer(
                assistant,
                add_special_tokens=False,
                truncation=True,
                max_length=max_response_tokens,
            ).input_ids
            is_truncated = len(response_ids) == max_response_tokens
            truncated += int(is_truncated)
            response_ids = response_ids + [tokenizer.eos_token_id]

            records.append(
                {
                    "idx": len(records),
                    "source_row": row_index,
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "response_truncated": is_truncated,
                    "question_sha256": sha256_bytes(user.encode("utf-8")),
                    "question": user,
                }
            )
            if len(records) >= n_prompts:
                break
        if len(records) >= n_prompts:
            break

    if len(records) < n_prompts:
        raise RuntimeError(f"only {len(records)} usable prompts, wanted {n_prompts}")

    manifest = {
        "dataset": OT_DATASET,
        "revision": OT_REVISION,
        "parquet": OT_PARQUET,
        "license": "apache-2.0",
        "n_prompts": len(records),
        "source_rows_used": [r["source_row"] for r in records],
        "max_prompt_tokens": max_prompt_tokens,
        "max_response_tokens": max_response_tokens,
        "truncated_responses": truncated,
        "skipped_prompt_too_long": skipped_prompt,
        "prompt_token_stats": {
            "min": min(len(r["prompt_ids"]) for r in records),
            "max": max(len(r["prompt_ids"]) for r in records),
            "mean": sum(len(r["prompt_ids"]) for r in records) / len(records),
        },
        "prompt_sha256": sha256_json([r["prompt_ids"] for r in records]),
        "response_sha256": sha256_json([r["response_ids"] for r in records]),
    }
    return records, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--n-eval-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--eval-docs", type=int, default=800)
    parser.add_argument("--n-prompts", type=int, default=3000)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-response-tokens", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.out_dir)
    cache = root / "cache"
    manifests = root / "manifests"
    (cache / "eval").mkdir(parents=True, exist_ok=True)
    (cache / "prompts").mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)

    eval_ids, fineweb_manifest = build_fineweb_eval(
        tokenizer, args.n_eval_blocks, args.block_size, args.eval_docs
    )
    np.save(cache / "eval" / "fineweb_eval_ids.npy", eval_ids)

    parquet_path = cache / "openthoughts" / OT_PARQUET
    records, ot_manifest = build_prompt_pool(
        tokenizer,
        parquet_path,
        args.n_prompts,
        args.max_prompt_tokens,
        args.max_response_tokens,
    )
    with (cache / "prompts" / "openthoughts_prompts.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {
            "model": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "vocab_size": int(tokenizer.vocab_size),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "chat_template_sha256": sha256_bytes((tokenizer.chat_template or "").encode()),
        },
        "fineweb_eval": fineweb_manifest,
        "openthoughts_prompts": ot_manifest,
    }
    (manifests / "gate1_data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({"fineweb_eval": fineweb_manifest, "openthoughts": ot_manifest}, indent=2))


if __name__ == "__main__":
    main()
