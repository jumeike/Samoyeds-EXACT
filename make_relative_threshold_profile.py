#!/usr/bin/env python3
"""
Generate a profile using a relative hotness threshold per layer.
An expert is hot if activation_rate >= (1 / num_experts) * (1 + delta).
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--delta", type=float, required=True,
                        help="Relative threshold over uniform rate. e.g., 1.0 => hot if >= 2x uniform.")
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

        # Determine number of experts in this layer
        expert_ids = [int(k) for k in stats.keys()]
        if not expert_ids:
            continue
        num_experts = len(expert_ids)
        uniform_rate = 1.0 / num_experts
        threshold = uniform_rate * (1.0 + args.delta)

        hot = []
        cold = []
        for k, v in stats.items():
            val = float(v.get(args.metric, 0.0))
            if val >= threshold:
                hot.append(int(k))
            else:
                cold.append(int(k))

        layer_data["classification"] = {"hot": sorted(hot), "cold": sorted(cold)}

    out_path = args.output_json
    if out_path is None:
        in_path = Path(args.input_json)
        out_dir = in_path.parent / "relative_threshold"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = in_path.stem
        out_path = str(out_dir / f"{stem}_delta{args.delta:.3f}.json")

    with open(out_path, "w") as f:
        json.dump(prof, f, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
