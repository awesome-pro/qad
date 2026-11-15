"""Compare two matched Q1/Q2 run manifests for config and prompt alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_qad.matching import compare_matched_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1", required=True)
    parser.add_argument("--q2", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    result = compare_matched_runs(
        json.loads(Path(args.q1).read_text()), json.loads(Path(args.q2).read_text())
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    raise SystemExit(0 if result["matched"] else 1)


if __name__ == "__main__":
    main()
