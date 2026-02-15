#!/usr/bin/env python3
"""
Generate a SparseML recipe from an EXACT profiling json (hot/cold experts).
This emits two pruning modifiers:
  - hot experts: 2:4 (OBS 2:4 pairwise)
  - cold experts: configurable mask_type (default: 1:2:4) with target sparsity
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict


SUPPORTED_MASKS = {
    "BF16OBS24pairPruningModifier": {
        "2:16", "3:16", "4:16", "5:16", "6:16", "7:16", "8:16",
        "9:16", "10:16", "11:16", "12:16",
        "6:8", "5:8", "4:8", "3:8", "2:8",
        "unstructured",
    },
    "BF16OBSTransnmvpairPruningModifier": {
        "1:2:4", "1:2:16", "1:2:32", "1:2:64", "1:2:128",
        "4:8:4", "4:8:16", "4:8:64", "4:8:128",
        "8:16:4", "8:16:16", "8:16:64", "8:16:128",
        "unstructured",
    },
}


def _validate_mask(modifier_name: str, mask_type: str) -> None:
    supported = SUPPORTED_MASKS.get(modifier_name, set())
    if mask_type not in supported:
        raise ValueError(
            f"{mask_type} mask_type not supported for {modifier_name}. "
            f"Supported: {sorted(supported)}"
        )


def _regex_for_layer(layer_idx: int, expert_ids: List[int]) -> str:
    if not expert_ids:
        return ""
    expert_re = "|".join(str(e) for e in sorted(expert_ids))
    return (
        rf"re:.*(?:_fsdp_wrapped_module\.)?(?:module\.)?(?:model\.)?layers\.{layer_idx}\."
        r"(?:_fsdp_wrapped_module\.)?mlp\.experts\."
        rf"({expert_re})\.(gate_proj|up_proj|down_proj)\.weight"
    )


def build_regex_list(layers: Dict, key: str) -> List[str]:
    regexes = []
    for layer_idx_str, layer_data in layers.items():
        try:
            layer_idx = int(layer_idx_str)
        except ValueError:
            continue
        cls = layer_data.get("classification", {})
        expert_ids = cls.get(key, [])
        if not expert_ids:
            continue
        rx = _regex_for_layer(layer_idx, expert_ids)
        if rx:
            regexes.append(rx)
    return regexes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiling_json", required=True)
    parser.add_argument("--out_yaml", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr_init", type=float, default=1e-5)
    parser.add_argument("--lr_final", type=float, default=1e-6)
    parser.add_argument("--hot_modifier", default="BF16OBS24pairPruningModifier",
                        choices=list(SUPPORTED_MASKS.keys()),
                        help="Modifier for hot experts.")
    parser.add_argument("--cold_modifier", default="BF16OBSTransnmvpairPruningModifier",
                        choices=list(SUPPORTED_MASKS.keys()),
                        help="Modifier for cold experts.")
    parser.add_argument("--hot_mask_type", default="2:8",
                        help="Mask type for hot experts.")
    parser.add_argument("--cold_mask_type", default="1:2:4",
                        help="Mask type for cold experts.")
    parser.add_argument("--cold_sparsity", type=float, default=0.75)
    parser.add_argument("--hot_sparsity", type=float, default=0.50)
    parser.add_argument("--start_epoch", type=float, default=0.1)
    parser.add_argument("--end_epoch", type=float, default=0.1)
    parser.add_argument("--num_grads", type=int, default=512)
    parser.add_argument("--fisher_block_size", type=int, default=32)
    args = parser.parse_args()

    _validate_mask(args.hot_modifier, args.hot_mask_type)
    _validate_mask(args.cold_modifier, args.cold_mask_type)

    with open(args.profiling_json) as f:
        prof = json.load(f)
    layers = prof.get("layers", {})

    hot_regexes = build_regex_list(layers, "hot")
    cold_regexes = build_regex_list(layers, "cold")

    out = Path(args.out_yaml)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("modifiers:")
    lines.append("  - !EpochRangeModifier")
    lines.append(f"    start_epoch: 0")
    lines.append(f"    end_epoch: {args.epochs}")
    lines.append("")
    lines.append("training_modifiers:")
    lines.append("  - !LearningRateFunctionModifier")
    lines.append("    start_epoch: 0.0")
    lines.append(f"    end_epoch: {min(args.epochs, 1.0)}")
    lines.append("    lr_func: linear")
    lines.append(f"    init_lr: {args.lr_init}")
    lines.append(f"    final_lr: {args.lr_final}")
    lines.append("")
    if hot_regexes:
        lines.append(f"  - !{args.hot_modifier}")
        lines.append("    params: [")
        for rx in hot_regexes:
            lines.append(f"      '{rx}',")
        lines.append("    ]")
        lines.append(f"    init_sparsity: {args.hot_sparsity}")
        lines.append(f"    final_sparsity: {args.hot_sparsity}")
        lines.append(f"    start_epoch: {args.start_epoch}")
        lines.append(f"    end_epoch: {args.end_epoch}")
        lines.append("    update_frequency: 1.0")
        lines.append("    inter_func: linear")
        lines.append("    global_sparsity: False")
        lines.append(f"    mask_type: '{args.hot_mask_type}'")
        lines.append(f"    num_grads: {args.num_grads}")
        lines.append("    damp: 1e-7")
        lines.append(f"    fisher_block_size: {args.fisher_block_size}")
        lines.append("    grad_sampler_kwargs:")
        lines.append("     batch_size: 16")
        lines.append("")
    if cold_regexes:
        lines.append(f"  - !{args.cold_modifier}")
        lines.append("    params: [")
        for rx in cold_regexes:
            lines.append(f"      '{rx}',")
        lines.append("    ]")
        lines.append(f"    init_sparsity: {args.cold_sparsity}")
        lines.append(f"    final_sparsity: {args.cold_sparsity}")
        lines.append(f"    start_epoch: {args.start_epoch}")
        lines.append(f"    end_epoch: {args.end_epoch}")
        lines.append("    update_frequency: 1.0")
        lines.append("    inter_func: linear")
        lines.append("    global_sparsity: False")
        lines.append(f"    mask_type: '{args.cold_mask_type}'")
        lines.append(f"    num_grads: {args.num_grads}")
        lines.append("    damp: 1e-7")
        lines.append(f"    fisher_block_size: {args.fisher_block_size}")
        lines.append("    grad_sampler_kwargs:")
        lines.append("     batch_size: 16")

    out.write_text("\n".join(lines))
    print(f"Wrote recipe: {out}")
    print(f"Hot regexes: {len(hot_regexes)}, Cold regexes: {len(cold_regexes)}")


if __name__ == "__main__":
    main()
