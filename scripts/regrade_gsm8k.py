"""Offline regrade of saved GSM8K generations with the current strict parser.

The original .json/.generations.jsonl files are left untouched. New files are
written with a ``.regraded`` suffix; per-example full text/token ids are copied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_qad.answer_parsing import PARSER_VERSION, parse_final_answer
from qwen_qad.benchmark_metrics import summarize_records
from qwen_qad.checkpoint import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--mode", choices=["nonthinking", "thinking"], required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    records = []
    with Path(args.generations).open() as handle:
        for line in handle:
            records.append(json.loads(line))
    if not records:
        raise SystemExit("no records")

    old_summary = json.loads(Path(args.summary).read_text()) if args.summary else None
    changed = []
    for index, record in enumerate(records):
        text = record.get("parse_text")
        if text is None:
            text = record.get("full_text", "")
        parsed = parse_final_answer(text, thinking_mode=args.mode == "thinking")
        gold = record["gold_answer"]
        correct = parsed.status == "ok" and parsed.value is not None and parsed.value == gold
        old_status = record.get("parse_status")
        old_value = record.get("parsed_value")
        if parsed.status != old_status or parsed.value != old_value:
            changed.append(
                {
                    "example_id": record.get("example_id", index),
                    "old_status": old_status,
                    "new_status": parsed.status,
                    "old_value": old_value,
                    "new_value": parsed.value,
                    "final_line": parsed.final_line[:200],
                }
            )
        record.update(
            {
                "parse_status": parsed.status,
                "parsed_value": parsed.value,
                "parse_marker": parsed.marker,
                "answer_region": parsed.answer_region,
                "final_line": parsed.final_line,
                "correct": correct,
            }
        )

    metrics = summarize_records(records, thinking=args.mode == "thinking")
    payload = {
        "regraded_from": str(args.generations),
        "source_sha256": file_sha256(args.generations),
        "parser_version": PARSER_VERSION,
        "mode": args.mode,
        "metrics": metrics,
        "old_metrics": old_summary["metrics"] if old_summary else None,
        "changed_examples": changed,
        "changed_count": len(changed),
    }
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    with out_prefix.with_suffix(".generations.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({"changed_count": len(changed), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
