"""Theoretical storage table for the pilot (no packed inference claim)."""

from __future__ import annotations

import json
from pathlib import Path

from qwen_qad.quantization import QuantConfig, storage_bits


def table_for(inventory_path: Path) -> list[dict]:
    inventory = json.loads(inventory_path.read_text())
    modules = [m for m in inventory["linear_modules"]["modules"] if m["quantizable"]]
    total_params = inventory["totals"]["parameters"]
    quantized_params = sum(m["parameters"] for m in modules)
    unquantized_params = total_params - quantized_params
    rows = []
    for name, cfg in (
        ("BF16", None),
        ("W4A16-G128", QuantConfig(bits=4, group_size=128)),
        ("W3A16-G32", QuantConfig(bits=3, group_size=32)),
        ("W3A16-G128", QuantConfig(bits=3, group_size=128)),
    ):
        if cfg is None:
            total_bits = total_params * 16
            payload_bits = quantized_params * 16
            metadata_bits = 0
        else:
            bits = [storage_bits(cfg, tuple(m["shape"])) for m in modules]
            payload_bits = sum(item["payload_bits"] for item in bits)
            metadata_bits = sum(item["total_bits"] - item["payload_bits"] for item in bits)
            total_bits = payload_bits + metadata_bits + unquantized_params * 16
        rows.append(
            {
                "model": inventory["model_id"],
                "config": name,
                "total_params": total_params,
                "quantized_params": quantized_params,
                "unquantized_params": unquantized_params,
                "payload_bits": payload_bits,
                "metadata_bits": metadata_bits,
                "total_bits": total_bits,
                "payload_mib": payload_bits / 8 / 2**20,
                "metadata_mib": metadata_bits / 8 / 2**20,
                "total_gib": total_bits / 8 / 2**30,
                "effective_bits_per_weight": total_bits / total_params,
            }
        )
    bf16 = rows[0]["total_gib"]
    for row in rows:
        row["compression_vs_bf16"] = bf16 / row["total_gib"]
    return rows


def main() -> None:
    inventories = [Path("experiments/gate1/inventory_qwen3_0p6b.json")]
    one_b = Path("experiments/gate7/inventory_cuda.json")
    if one_b.exists():
        inventories.append(one_b)
    all_rows = []
    for path in inventories:
        if path.exists():
            all_rows.extend(table_for(path))

    lines = [
        "# Theoretical storage accounting",
        "",
        "These are **theoretical packed-model sizes** computed from parameter shapes "
        "and the declared quantizer metadata. They are not measured packed "
        "artifacts and do not imply inference speedups. Measured training memory is "
        "reported separately in `reports/findings.md`.",
        "",
        "| model | config | total params | quantized params | payload MiB | metadata MiB | total GiB | eff bits/weight | vs BF16 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        lines.append(
            "| {model} | {config} | {total_params:,} | {quantized_params:,} | "
            "{payload_mib:.1f} | {metadata_mib:.1f} | {total_gib:.2f} | "
            "{effective_bits_per_weight:.3f} | {compression_vs_bf16:.2f}x |".format(**row)
        )
    lines.extend(
        [
            "",
            "Metadata assumes one 16-bit scale per group and no zero-point "
            "(symmetric scheme); alignment/packing overhead is unknown and excluded.",
            "Unquantized parameters (embedding, tied LM head, norms) are counted at 16 bits.",
        ]
    )
    Path("reports/memory_accounting.md").write_text("\n".join(lines) + "\n")
    Path("reports/memory_accounting.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
