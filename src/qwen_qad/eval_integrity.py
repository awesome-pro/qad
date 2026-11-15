"""Integrity checks across strict-evaluation summaries and saved records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .benchmark_compare import paired_compare
from .benchmark_metrics import summarize_records

METRIC_KEYS = (
    "accuracy",
    "correct",
    "count",
    "marker_present_rate",
    "terminal_marker_present_rate",
    "parsed_answer_rate",
    "completion_rate",
    "truncation_rate",
    "thinking_closed_rate",
    "mean_response_tokens",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def check_records(records: Sequence[dict]) -> list[str]:
    problems = []
    ids = [r.get("example_id") for r in records]
    if len(ids) != len(set(ids)):
        problems.append("duplicate example_id values")
    for record in records:
        expected = bool(
            record.get("parse_status") == "ok"
            and record.get("parsed_value") is not None
            and record.get("parsed_value") == record.get("gold_answer")
        )
        if bool(record.get("correct")) != expected:
            problems.append(f"correct flag mismatch for {record.get('example_id')}")
        if record.get("parse_status") not in {
            "ok", "missing", "ambiguous", "malformed", "incomplete_thinking"
        }:
            problems.append(f"bad status for {record.get('example_id')}")
    return problems


def check_summary(summary: dict, records: Sequence[dict], mode: str) -> list[str]:
    problems = []
    recomputed = summarize_records(records, thinking=mode == "thinking")
    for key in METRIC_KEYS:
        if key not in summary.get("metrics", {}):
            problems.append(f"summary missing metric {key}")
        elif isinstance(recomputed[key], float):
            if not _close(summary["metrics"][key], recomputed[key]):
                problems.append(f"metric {key} disagrees with records")
        elif summary["metrics"][key] != recomputed[key]:
            problems.append(f"metric {key} disagrees with records")
    if summary["metrics"].get("parse_status_counts") != recomputed["parse_status_counts"]:
        problems.append("parse_status_counts disagree with records")
    return problems


def check_provenance(provenance: dict, summary: dict, run: dict) -> list[str]:
    problems = []
    if provenance.get("metrics") != summary.get("metrics"):
        problems.append("provenance metrics differ from canonical summary")
    if provenance.get("parser_version") != summary.get("parser_version"):
        problems.append("provenance parser_version differs from canonical")
    if provenance.get("source_sha256") != summary.get("source_sha256"):
        problems.append("provenance source hash differs from canonical")
    checkpoint = provenance.get("checkpoint") or {}
    expected = run.get("best_checkpoint", {}).get("sha256")
    if not expected or checkpoint.get("sha256") != expected:
        problems.append("provenance checkpoint SHA does not match run manifest")
    return problems


def check_paired(
    paired: dict,
    q1_records: Sequence[dict],
    q2_records: Sequence[dict],
    q1_summary: dict,
    q2_summary: dict,
) -> list[str]:
    problems = []
    expected = paired_compare(q1_records, q2_records)
    for key in ("q1_accuracy", "q2_accuracy", "difference_q2_minus_q1", "mcnemar_p"):
        if not _close(paired.get(key, -999), expected[key]):
            problems.append(f"paired {key} disagrees with records")
    if paired.get("paired_counts") != expected["paired_counts"]:
        problems.append("paired counts disagree with records")
    if paired.get("parser_versions") != [
        q1_summary.get("parser_version"),
        q2_summary.get("parser_version"),
    ]:
        problems.append("paired parser_versions disagree with summaries")
    if paired.get("q1_source_sha256") != q1_summary.get("source_sha256"):
        problems.append("paired Q1 source hash disagrees")
    if paired.get("q2_source_sha256") != q2_summary.get("source_sha256"):
        problems.append("paired Q2 source hash disagrees")
    return problems


def run_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    results = []
    for entry in manifest["entries"]:
        name = entry["name"]
        records = load_jsonl(entry["generations"])
        summary = load_json(entry["summary"])
        problems = check_records(records)
        problems += check_summary(summary, records, entry["mode"])
        if summary.get("parser_version") != manifest["parser_version"]:
            problems.append("summary parser_version differs from frozen manifest")
        source_path = summary.get("regraded_from")
        if not source_path or not Path(source_path).exists():
            problems.append("summary regraded_from source is missing")
        elif summary.get("source_sha256") != file_sha256(source_path):
            problems.append("summary source_sha256 differs from original generation source")
        if entry.get("provenance"):
            provenance = load_json(entry["provenance"])
            problems += check_provenance(
                provenance, summary, load_json(entry["run_json"])
            )
        results.append({"name": name, "passed": not problems, "problems": problems})
    for entry in manifest.get("paired", []):
        name = entry["name"]
        paired = load_json(entry["summary"])
        problems = check_paired(
            paired,
            load_jsonl(entry["q1_generations"]),
            load_jsonl(entry["q2_generations"]),
            load_json(entry["q1_summary"]),
            load_json(entry["q2_summary"]),
        )
        if paired.get("parser_versions") != [manifest["parser_version"]] * 2:
            problems.append("paired parser version differs from frozen manifest")
        results.append({"name": name, "passed": not problems, "problems": problems})
    return {
        "manifest": str(manifest_path),
        "parser_version": manifest["parser_version"],
        "passed": all(item["passed"] for item in results),
        "results": results,
    }
