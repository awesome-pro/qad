"""Per-example diff between two regrade generations files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: str) -> dict[str, dict]:
    records = {}
    with Path(path).open() as handle:
        for line in handle:
            record = json.loads(line)
            records[record["example_id"]] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    old = load(args.old)
    new = load(args.new)
    if set(old) != set(new):
        raise SystemExit("example ids differ")
    changes = []
    transitions = Counter()
    for example_id in old:
        a, b = old[example_id], new[example_id]
        if (
            a.get("parse_status") != b.get("parse_status")
            or a.get("parsed_value") != b.get("parsed_value")
            or bool(a.get("correct")) != bool(b.get("correct"))
        ):
            transitions[f"{a.get('parse_status')}->{b.get('parse_status')}"] += 1
            changes.append(
                {
                    "example_id": example_id,
                    "old_status": a.get("parse_status"),
                    "new_status": b.get("parse_status"),
                    "old_value": a.get("parsed_value"),
                    "new_value": b.get("parsed_value"),
                    "gold": b.get("gold_answer"),
                    "old_correct": bool(a.get("correct")),
                    "new_correct": bool(b.get("correct")),
                    "terminal_line": b.get("final_line", "")[:200],
                }
            )
    payload = {
        "old": args.old,
        "new": args.new,
        "changed_count": len(changes),
        "transitions": dict(transitions),
        "changes": changes,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"changed_count": len(changes), "transitions": dict(transitions)}, indent=2))


if __name__ == "__main__":
    main()
