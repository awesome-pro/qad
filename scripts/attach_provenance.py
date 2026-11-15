"""Attach checkpoint provenance from a matched run manifest to an eval artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--run-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    artifact = json.loads(Path(args.artifact).read_text())
    run = json.loads(Path(args.run_json).read_text())
    best = run["best_checkpoint"]
    artifact["checkpoint"] = {
        "path": best["path"],
        "sha256": best["sha256"],
        "step": best["step"],
        "val_nll": best["val_nll"],
        "inference_weights": best.get("inference_weights"),
        "inference_sha256": best.get("inference_sha256"),
        "quantization": run["quantization"],
        "model": run["model"],
    }
    Path(args.out).write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"{args.artifact} -> {args.out}: {best['sha256'][:16]}")


if __name__ == "__main__":
    main()
