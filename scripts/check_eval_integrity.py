"""Check summaries, provenance, paired outputs and records for consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_qad.eval_integrity import run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_manifest(args.manifest)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "parser_version": result["parser_version"],
                      "results": result["results"]}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
