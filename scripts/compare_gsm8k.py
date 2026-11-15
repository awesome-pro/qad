"""Paired comparison of two strict GSM8K evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_qad.benchmark_compare import paired_compare


def load_records(path: str) -> list[dict]:
    records = []
    with Path(path).open() as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1", required=True, help="Q1 .generations.jsonl")
    parser.add_argument("--q2", required=True, help="Q2 .generations.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--q1-summary", default=None)
    parser.add_argument("--q2-summary", default=None)
    args = parser.parse_args()
    result = paired_compare(load_records(args.q1), load_records(args.q2))
    if args.q1_summary and args.q2_summary:
        s1 = json.loads(Path(args.q1_summary).read_text())
        s2 = json.loads(Path(args.q2_summary).read_text())
        result["parser_versions"] = [s1.get("parser_version"), s2.get("parser_version")]
        result["q1_source_sha256"] = s1.get("source_sha256")
        result["q2_source_sha256"] = s2.get("source_sha256")
        result["canonical_summaries"] = [args.q1_summary, args.q2_summary]
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
