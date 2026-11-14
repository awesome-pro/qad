"""Gate 1 local smoke test: T0 NLL, teacher self-KL and chat generation.

This is a correctness/plumbing check on the Mac, not a reported baseline. The
official T0 numbers are produced on the GPU in the first paid session.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qwen_qad.losses import (
    chunked_forward_kl,
    chunked_nll,
    gather_scored_tokens,
    next_token_score_mask,
)
from qwen_qad.models import load_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--eval", default="data/cache/eval/fineweb_eval_ids.npy")
    parser.add_argument("--prompts", default="data/cache/prompts/openthoughts_prompts.jsonl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--quantize", action="store_true", help="evaluate Q0 (W3A16 RTN)")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tokenizer = load_tokenizer(args.model, args.revision)
    model = load_model(args.model, args.revision, device=args.device, dtype=dtype)
    label = "Q0" if args.quantize else "T0"
    if args.quantize:
        from qwen_qad.models import set_trainable_quantized_only, wrap_quantized_linears
        from qwen_qad.quantization import QuantConfig

        replaced = wrap_quantized_linears(
            model, QuantConfig(bits=args.bits, group_size=args.group_size)
        )
        trainable = set_trainable_quantized_only(model)
        print(f"wrapped {len(replaced)} linear modules; trainable params {trainable:,}")

    lm_weight = model.get_output_embeddings().weight.detach()
    print(f"loaded {args.model} on {args.device} {dtype}; "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    blocks = np.load(args.eval)
    input_ids = torch.tensor(blocks[: args.batch_size], dtype=torch.long, device=args.device)
    seq_len = input_ids.shape[1]
    prompt_lens = torch.ones(input_ids.shape[0], dtype=torch.long, device=args.device)
    response_lens = torch.full_like(prompt_lens, seq_len - 1)
    mask = next_token_score_mask(prompt_lens, response_lens, seq_len, device=args.device)

    start = time.perf_counter()
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        h_scored, targets = gather_scored_tokens(hidden, input_ids, mask)
        nll = chunked_nll(h_scored, targets, lm_weight, chunk_size=256)
        self_kl = chunked_forward_kl(h_scored, h_scored, lm_weight, chunk_size=256)
    elapsed = time.perf_counter() - start

    n_tokens = h_scored.shape[0]
    print(f"scored positions   {n_tokens}")
    print(f"{label} NLL             {nll.item():.6f}  (ppl {torch.exp(nll).item():.4f})")
    print(f"teacher self-KL    {self_kl.item():.3e}")
    print(f"forward latency    {elapsed:.2f}s  ({n_tokens / elapsed:.0f} tok/s, one pass)")

    # Cross-check the chunked path against a naive full-logits computation on
    # the first sequence. Memory is fine at 1x1024 on this machine.
    one = input_ids[:1]
    mask_one = next_token_score_mask(
        prompt_lens[:1], response_lens[:1], seq_len, device=args.device
    )
    with torch.no_grad():
        hidden_one = model.model(input_ids=one, use_cache=False).last_hidden_state
        h_one, t_one = gather_scored_tokens(hidden_one, one, mask_one)
        nll_one = chunked_nll(h_one, t_one, lm_weight, chunk_size=256)
        logits = model(input_ids=one).logits.float()
        naive = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            one[:, 1:].reshape(-1),
        )
    print(f"single-sequence NLL  chunked {nll_one.item():.6f} vs naive "
          f"{naive.item():.6f} (difference {abs(nll_one.item() - naive.item()):.2e})")

    # Chat-template generation smoke test.
    with Path(args.prompts).open() as handle:
        record = json.loads(handle.readline())
    prompt_ids = torch.tensor([record["prompt_ids"]], dtype=torch.long, device=args.device)
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=args.new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_elapsed = time.perf_counter() - start
    continuation = generated[0, prompt_ids.shape[1] :]
    print(f"\ngeneration: {continuation.shape[0]} tokens in {gen_elapsed:.1f}s")
    print("-" * 70)
    print(tokenizer.decode(continuation, skip_special_tokens=False)[:1200])
    print("-" * 70)

    if args.device == "mps" and hasattr(torch, "mps"):
        print(f"mps allocated: {torch.mps.current_allocated_memory() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
