"""Verify two Qwen tokenizers are compatible before a cross-size handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

SAMPLES = [
    "<think>", "</think>", "#### 42", "Answer:", "$233.00",
    "Janet's ducks lay 16 eggs per day.", "é—ü—汉",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--a-revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--b", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--b-revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    a = AutoTokenizer.from_pretrained(args.a, revision=args.a_revision)
    b = AutoTokenizer.from_pretrained(args.b, revision=args.b_revision)
    vocab_a, vocab_b = a.get_vocab(), b.get_vocab()
    checks = {
        "vocab_size_equal": len(vocab_a) == len(vocab_b),
        "vocab_dict_equal": vocab_a == vocab_b,
        "chat_template_equal": getattr(a, "chat_template", None) == getattr(b, "chat_template", None),
        "bos_equal": a.bos_token == b.bos_token,
        "eos_equal": a.eos_token == b.eos_token,
        "pad_equal": a.pad_token == b.pad_token,
        "special_ids_equal": {
            name: (a.convert_tokens_to_ids(name), b.convert_tokens_to_ids(name))
            for name in ("<think>", "</think>", "<|im_start|>", "<|im_end|>")
        },
        "sample_encodings_equal": {},
    }
    for sample in SAMPLES:
        ids_a = a.encode(sample, add_special_tokens=False)
        ids_b = b.encode(sample, add_special_tokens=False)
        checks["sample_encodings_equal"][sample] = ids_a == ids_b
    payload = {
        "tokenizer_a": {"id": args.a, "revision": args.a_revision, "vocab_size": len(vocab_a),
                        "chat_template_sha256": hashlib.sha256(str(a.chat_template).encode()).hexdigest()},
        "tokenizer_b": {"id": args.b, "revision": args.b_revision, "vocab_size": len(vocab_b),
                        "chat_template_sha256": hashlib.sha256(str(b.chat_template).encode()).hexdigest()},
        "checks": checks,
        "passed": bool(
            checks["vocab_size_equal"]
            and checks["vocab_dict_equal"]
            and checks["chat_template_equal"]
            and checks["eos_equal"]
            and checks["pad_equal"]
            and all(checks["sample_encodings_equal"].values())
            and all(x == y for x, y in checks["special_ids_equal"].values())
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
