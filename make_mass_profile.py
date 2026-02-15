#!/usr/bin/env python3
"""
Generate a hot/cold profile by cumulative routing mass per layer.
Hot experts are the smallest set whose cumulative activation_rate >= mass.
This keeps adaptivity without fixing a hot ratio.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--mass", type=float, required=True,
                        help="Cumulative mass to cover with hot experts (e.g., 0.5 or 0.7).")
    parser.add_argument("--metric", choices=["activation_rate", "count"], default="activation_rate")
    parser.add_argument("--min_hot", type=int, default=1, help="Minimum hot experts per layer.")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    if not (0.0 < args.mass <= 1.0):
        raise ValueError("--mass must be in (0, 1].")

    with open(args.input_json, "r") as f:
        prof = json.load(f)

    layers = prof.get("layers", {})
    if not layers:
        raise ValueError("No layers found in profile.")

    for layer_idx, layer_data in layers.items():
        stats = layer_data.get("expert_stats", {})
        if not stats:
            continue

        items = []
        for k, v in stats.items():
            items.append((int(k), float(v.get(args.metric, 0.0))))
        items.sort(key=lambda x: x[1], reverse=True)

        hot = []
        cold = []
        cum = 0.0
        for expert_id, val in items:
            if cum < args.mass or len(hot) < args.min_hot:
                hot.append(expert_id)
                cum += val
            else:
                cold.append(expert_id)

        layer_data["classification"] = {"hot": sorted(hot), "cold": sorted(cold)}

    out_path = args.output_json
    if out_path is None:
        in_path = Path(args.input_json)
        out_dir = in_path.parent / "mass_profile"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = in_path.stem
        out_path = str(out_dir / f"{stem}_mass{args.mass:.3f}.json")

    with open(out_path, "w") as f:
        json.dump(prof, f, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
