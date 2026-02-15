#!/usr/bin/env python3
"""
Compute compressed parameter storage for SpTC 2:4 / 1:4 formats.
Reports dense bytes vs compressed bytes for hot/cold policies.
"""
import argparse
import json
from transformers import AutoConfig


def sp_tc_bytes(num_params, value_ratio, meta_bits_per_group=4.0, group_size=4):
    # values stored in FP16
    values_bytes = num_params * value_ratio * 2
    # metadata stored per sparse group (default group size 4):
    # metadata_bytes = (#groups) * bits_per_group / 8
    meta_bytes = (num_params / group_size) * meta_bits_per_group / 8
    return values_bytes + meta_bytes


def _cfg_get_first(cfg, names):
    for n in names:
        if hasattr(cfg, n):
            v = getattr(cfg, n)
            if v is not None:
                return v
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen1.5-MoE-A2.7B')
    parser.add_argument('--profiling_json', required=True)
    parser.add_argument('--cold_values_as_24', action='store_true',
                        help='Ablation: treat cold values as K/2 (2:4-like) instead of compact K/4.')
    parser.add_argument('--cold_meta_compact', action='store_true',
                        help='Ablation: treat cold metadata as compact 1:4 (2 bits per 4-value group). '
                             'Default models current impl: expanded 2:4 metadata (4 bits per group).')
    parser.add_argument('--hot_meta_bits_per_group', type=float, default=4.0,
                        help='Metadata bits per 4-value group for hot experts (default: 4).')
    parser.add_argument('--group_size', type=int, default=4,
                        help='Sparse group size (default: 4).')
    args = parser.parse_args()

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    hidden = _cfg_get_first(cfg, ["hidden_size", "d_model"])
    interm = _cfg_get_first(
        cfg,
        [
            "moe_intermediate_size",   # Qwen/DeepSeek
            "intermediate_size",       # Mixtral
            "ffn_dim",                 # some MoE variants
        ],
    )

    if hidden is None or interm is None:
        raise ValueError(
            "Could not infer expert dimensions from config. "
            "Expected hidden_size/d_model and moe_intermediate_size/intermediate_size/ffn_dim."
        )

    # per expert params (gate, up, down)
    per_expert_params = 3 * hidden * interm

    with open(args.profiling_json) as f:
        prof = json.load(f)
    layers = prof.get('layers', {})
    hot = cold = total_instances = 0
    for _, layer_data in layers.items():
        cls = layer_data.get('classification', {})
        h = len(cls.get('hot', []))
        c = len(cls.get('cold', []))
        hot += h
        cold += c
        total_instances += (h + c)

    if total_instances == 0:
        raise ValueError("No classified experts found in profiling JSON.")

    total_expert_params = per_expert_params * total_instances
    dense_bytes = total_expert_params * 2
    total = hot + cold
    hot_ratio = hot / total if total else 0.0
    cold_ratio = cold / total if total else 0.0

    hot_bytes = sp_tc_bytes(
        total_expert_params * hot_ratio,
        0.5,
        meta_bits_per_group=args.hot_meta_bits_per_group,
        group_size=args.group_size,
    )
    # Current EXACT dual-storage:
    # - cold values: compact K/4 (ratio 0.25)
    # - cold metadata: expanded to 2:4-compatible domain => 4 bits per 4-value group
    cold_ratio_val = 0.5 if args.cold_values_as_24 else 0.25
    cold_meta_bits_per_group = 2.0 if args.cold_meta_compact else 4.0
    cold_bytes = sp_tc_bytes(
        total_expert_params * cold_ratio,
        cold_ratio_val,
        meta_bits_per_group=cold_meta_bits_per_group,
        group_size=args.group_size,
    )
    compressed_bytes = hot_bytes + cold_bytes

    print("SpTC PARAMETER STORAGE (experts only)")
    print(f"Model: {args.model}")
    print(f"Assumed expert matrices: gate/up/down, hidden={hidden}, intermediate={interm}")
    print(f"Expert instances in profiling JSON: {total_instances} (hot={hot}, cold={cold})")
    print(f"Assumptions:")
    print(f"  Group size: {args.group_size}")
    print(f"  Hot values ratio: 0.5 (K/2), hot metadata: {args.hot_meta_bits_per_group} bits/group")
    print(f"  Cold values ratio: {cold_ratio_val:.2f} "
          f"({'K/2' if args.cold_values_as_24 else 'K/4 compact'})")
    print(f"  Cold metadata: {cold_meta_bits_per_group} bits/group "
          f"({'compact 1:4' if args.cold_meta_compact else 'expanded 2:4-compatible'})")
    print("  Note: indices / non-expert params / runtime activations are excluded.")
    print(f"Dense bytes: {dense_bytes/1024**3:.3f} GB")
    print(f"Compressed bytes: {compressed_bytes/1024**3:.3f} GB")
    if compressed_bytes > 0:
        print(f"Compression ratio (dense/compressed): {dense_bytes/compressed_bytes:.3f}x")
    print(f"Compressed fraction (compressed/dense): {compressed_bytes/dense_bytes*100:.1f}%")
    saved = dense_bytes - compressed_bytes
    print(f"Saved: {saved/1024**3:.3f} GB ({saved/dense_bytes*100:.1f}%)")


if __name__ == "__main__":
    main()
