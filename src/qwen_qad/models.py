"""Model loading and checkpoint-derived inventory.

The inventory is the authoritative source of parameter counts and module
coverage (see plan.md section 5.2): architecture arithmetic only cross-checks
it. Quantization wrapping is not implemented here yet; Gate 2 adds it.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

QUANTIZABLE_PROJECTIONS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Architecture-derived expectations from plan.md section 5.2. A mismatch is a
# warning, not an error: the checkpoint-derived inventory always wins.
EXPECTED_PARAMETERS: dict[str, dict[str, int]] = {
    "Qwen/Qwen3-0.6B": {"total": 596_049_920, "quantizable": 440_401_920},
    "Qwen/Qwen3-1.7B": {"total": 1_720_574_976, "quantizable": 1_409_286_144},
}


def is_quantizable_name(module_name: str) -> bool:
    return module_name.rsplit(".", 1)[-1] in QUANTIZABLE_PROJECTIONS


def load_tokenizer(model_id: str, revision: str | None = None) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def load_model(
    model_id: str,
    revision: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PreTrainedModel:
    """Load a causal LM in the requested dtype/device.

    transformers 5 renamed ``torch_dtype`` to ``dtype``; try the new name first
    and fall back for older versions.
    """
    kwargs: dict[str, Any] = {"revision": revision, "attn_implementation": "sdpa"}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    model.to(device)
    model.eval()
    return model


def _category(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if name.startswith("lm_head"):
        return "lm_head"
    if name.endswith("norm.weight") or ".norm." in name:
        return "norm"
    if name.endswith(".bias"):
        return "bias"
    return "other"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_inventory(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase | None,
    model_id: str,
    revision: str | None,
) -> dict[str, Any]:
    linear_modules: list[dict[str, Any]] = []
    quantizable_params = 0
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantizable = is_quantizable_name(name)
            numel = int(module.weight.numel())
            if module.bias is not None:
                numel += int(module.bias.numel())
            linear_modules.append(
                {
                    "name": name,
                    "shape": [int(s) for s in module.weight.shape],
                    "parameters": numel,
                    "quantizable": quantizable,
                }
            )
            if quantizable:
                quantizable_params += int(module.weight.numel())

    categories: dict[str, int] = {}
    for name, param in model.named_parameters():
        total_params += int(param.numel())
        categories[_category(name)] = categories.get(_category(name), 0) + int(param.numel())

    # Tied embeddings: Qwen3 stores one tensor for input embedding and LM head.
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    tied = bool(
        input_emb is not None
        and output_emb is not None
        and input_emb.weight.data_ptr() == output_emb.weight.data_ptr()
    )

    warnings: list[str] = []
    expected = EXPECTED_PARAMETERS.get(model_id)
    if expected:
        if total_params != expected["total"]:
            warnings.append(
                f"total parameters {total_params:,} != architecture expectation "
                f"{expected['total']:,}"
            )
        if quantizable_params != expected["quantizable"]:
            warnings.append(
                f"quantizable parameters {quantizable_params:,} != architecture "
                f"expectation {expected['quantizable']:,}"
            )

    tokenizer_info: dict[str, Any] = {}
    if tokenizer is not None:
        template = getattr(tokenizer, "chat_template", None) or ""
        tokenizer_info = {
            "vocab_size": int(tokenizer.vocab_size),
            "model_max_length": int(tokenizer.model_max_length),
            "special_tokens": dict(tokenizer.special_tokens_map),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "chat_template_sha256": _sha256(template),
            "chat_template": template,
        }
        if tokenizer.pad_token_id is None:
            warnings.append("tokenizer has no pad token; training code must set one explicitly")

    config = model.config.to_dict()
    keep_config = {
        k: config.get(k)
        for k in (
            "architectures",
            "num_hidden_layers",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "tie_word_embeddings",
            "max_position_embeddings",
            "torch_dtype",
            "dtype",
        )
        if k in config
    }

    return {
        "model_id": model_id,
        "revision": revision,
        "config": keep_config,
        "totals": {
            "parameters": total_params,
            "quantizable_parameters": quantizable_params,
            "non_quantizable_parameters": total_params - quantizable_params,
            "categories": categories,
        },
        "linear_modules": {
            "count": len(linear_modules),
            "quantized_count": sum(1 for m in linear_modules if m["quantizable"]),
            "modules": linear_modules,
        },
        "tied_embeddings": tied,
        "tokenizer": tokenizer_info,
        "warnings": warnings,
    }


def summarize_inventory(inventory: dict[str, Any]) -> str:
    t = inventory["totals"]
    lm = inventory["linear_modules"]
    lines = [
        f"model             {inventory['model_id']} @ {inventory['revision']}",
        f"parameters        {t['parameters']:,} total",
        f"  quantizable     {t['quantizable_parameters']:,} "
        f"({t['quantizable_parameters'] / t['parameters']:.1%})",
        f"  frozen          {t['non_quantizable_parameters']:,}",
        f"linear modules    {lm['count']} ({lm['quantized_count']} quantizable)",
        f"tied embeddings   {inventory['tied_embeddings']}",
        f"categories        {t['categories']}",
    ]
    tok = inventory.get("tokenizer") or {}
    if tok:
        lines.append(
            f"tokenizer         vocab={tok['vocab_size']} eos={tok['eos_token_id']} "
            f"pad={tok['pad_token_id']} template_sha={tok['chat_template_sha256'][:12]}"
        )
    if inventory["warnings"]:
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in inventory["warnings"])
    return "\n".join(lines)


def _get_parent_module(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def wrap_quantized_linears(
    model: nn.Module,
    config: "QuantConfig",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Replace eligible ``nn.Linear`` modules with ``FakeQuantLinear``.

    Only the seven Qwen3 projection names are eligible. ``include``/``exclude``
    are full-match regular expressions over module names. Returns the replaced
    names. Master weights are fp32 on the source module's device.
    """
    import re

    from .quantization import FakeQuantLinear

    include_re = [re.compile(p) for p in include] if include else []
    exclude_re = [re.compile(p) for p in exclude] if exclude else []

    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not is_quantizable_name(name):
            continue
        if include_re and not any(pattern.fullmatch(name) for pattern in include_re):
            continue
        if exclude_re and any(pattern.fullmatch(name) for pattern in exclude_re):
            continue
        parent, attribute = _get_parent_module(model, name)
        replacement = FakeQuantLinear.from_linear(module, config, name=name)
        setattr(parent, attribute, replacement)
        replaced.append(name)
    return replaced


def set_trainable_quantized_only(model: nn.Module) -> int:
    """Freeze every parameter except the fake-quantized master weights.

    Returns the number of trainable parameters. Call after wrapping: the
    embeddings, tied LM head, and all norms become frozen.
    """
    from .quantization import FakeQuantLinear

    master_ids = {
        id(module.weight) for module in model.modules() if isinstance(module, FakeQuantLinear)
    }
    trainable = 0
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in master_ids)
        if parameter.requires_grad:
            trainable += int(parameter.numel())
    return trainable


def count_fake_quant_modules(model: nn.Module) -> int:
    from .quantization import FakeQuantLinear

    return sum(1 for module in model.modules() if isinstance(module, FakeQuantLinear))
