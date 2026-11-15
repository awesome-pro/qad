"""Create disjoint validation and final-test token blocks.

Validation: first 128 blocks of the original FineWeb-Edu held-out file (the
protocol's inspected subset; now explicitly validation-only).
Final test: a newly prepared WikiText-103 test stream (disjoint corpus), 128
blocks of 1,024 tokens, pinned dataset revision and license.

The original `fineweb_eval_ids.npy` is left untouched as a historical artifact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

TOKENIZER_ID = "Qwen/Qwen3-0.6B"
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
WIKITEXT_ID = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_LICENSE = "cc-by-sa-3.0 / gfdl"
VAL_BLOCKS = 128
TEST_BLOCKS = 128
BLOCK_SIZE = 1024


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def block_hashes(array: np.ndarray) -> set[str]:
    return {hashlib.sha256(block.tobytes()).hexdigest() for block in array}


def main() -> None:
    root = Path("data")
    eval_dir = root / "cache" / "eval"
    manifests = root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    old = np.load(eval_dir / "fineweb_eval_ids.npy")
    if old.shape[1] != BLOCK_SIZE:
        raise RuntimeError(f"unexpected block size {old.shape}")
    validation = old[:VAL_BLOCKS].astype(np.int32)
    np.save(eval_dir / "fineweb_val_ids.npy", validation)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    parquet = hf_hub_download(
        WIKITEXT_ID,
        "wikitext-103-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset",
        revision=WIKITEXT_REVISION,
        local_dir=root / "cache" / "wikitext",
    )
    import pyarrow.parquet as pq

    rows = pq.ParquetFile(parquet).read().to_pylist()
    flat: list[int] = []
    eos = tokenizer.eos_token_id
    for row in rows:
        text = row["text"].strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        flat.extend(ids)
        flat.append(eos)
    needed = TEST_BLOCKS * BLOCK_SIZE
    if len(flat) < needed:
        raise RuntimeError(f"WikiText test has only {len(flat)} tokens, need {needed}")
    test = np.asarray(flat[:needed], dtype=np.int32).reshape(TEST_BLOCKS, BLOCK_SIZE)
    np.save(eval_dir / "wikitext_test_ids.npy", test)

    overlap = block_hashes(validation) & block_hashes(test)
    if overlap:
        raise RuntimeError(f"validation/test block overlap: {len(overlap)} blocks")

    info = HfApi().dataset_info(WIKITEXT_ID, revision=WIKITEXT_REVISION)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION},
        "validation": {
            "source": "data/cache/eval/fineweb_eval_ids.npy (first 128 blocks)",
            "original_file_sha256": sha256_array(old),
            "array_sha256": sha256_array(validation),
            "blocks": int(validation.shape[0]),
            "block_size": BLOCK_SIZE,
            "tokens": int(validation.size),
            "license": "odc-by",
        },
        "test": {
            "dataset": WIKITEXT_ID,
            "config": WIKITEXT_CONFIG,
            "split": "test",
            "revision": WIKITEXT_REVISION,
            "resolved_sha": info.sha,
            "license": WIKITEXT_LICENSE,
            "array_sha256": sha256_array(test),
            "blocks": int(test.shape[0]),
            "block_size": BLOCK_SIZE,
            "tokens": int(test.size),
            "separator_token_id": int(eos),
        },
        "overlap": {
            "block_hashes_intersection": len(overlap),
            "validation_first_block_sha256": hashlib.sha256(
                validation[0].tobytes()
            ).hexdigest(),
            "test_first_block_sha256": hashlib.sha256(test[0].tobytes()).hexdigest(),
        },
        "notes": [
            "Validation and final test are disjoint at the corpus and block level.",
            "FineWeb-Edu metrics computed on the original 256-block file are historical and must be labelled as the original protocol.",
            "WikiText-103 is a public benchmark; contamination in pretraining is possible and is stated as a limitation.",
        ],
    }
    (manifests / "gate8_splits_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
