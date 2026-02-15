#!/usr/bin/env python3
"""
Generate a fixed hot-ratio profile from an existing profiling JSON.
Selects top-K% experts per layer by activation_rate (or count).
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--hot_ratio", type=float, required=True, help="Fraction of experts to mark hot (e.g., 0.25)")
    parser.add_argument("--metric", choices=["activation_rate", "count"], default="activation_rate")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    with open(args.input_json, "r") as f:
        prof = json.load(f)

    layers = prof.get("layers", {})
    if not layers:
        raise ValueError("No layers found in profile.")

    for layer_idx, layer_data in layers.items():
        stats = layer_data.get("expert_stats", {})
        if not stats:
            continue
        # Build list of (expert_id, metric_value)
        items = []
        for k, v in stats.items():
            items.append((int(k), float(v.get(args.metric, 0.0))))
        items.sort(key=lambda x: x[1], reverse=True)
        num_experts = len(items)
        k_hot = max(1, int(round(num_experts * args.hot_ratio)))
        hot = sorted([eid for eid, _ in items[:k_hot]])
        cold = sorted([eid for eid, _ in items[k_hot:]])
        layer_data["classification"] = {"hot": hot, "cold": cold}

    out_path = args.output_json
    if out_path is None:
        in_path = Path(args.input_json)
        out_dir = in_path.parent / "fixed_ratio"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = in_path.stem
        out_path = str(out_dir / f"{stem}_ratio{args.hot_ratio:.3f}.json")
    with open(out_path, "w") as f:
        json.dump(prof, f, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
