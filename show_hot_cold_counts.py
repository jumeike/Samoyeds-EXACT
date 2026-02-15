#!/usr/bin/env python3
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Show per-layer and total hot/cold counts from a profiling JSON.")
    parser.add_argument("--profiling_json", required=True, help="Path to profiling JSON")
    args = parser.parse_args()

    with open(args.profiling_json, "r") as f:
        prof = json.load(f)

    layers = prof.get("layers", {})
    if not layers:
        raise ValueError("No 'layers' found in profiling JSON.")

    total_hot = 0
    total_cold = 0
    print(f"profiling_json: {args.profiling_json}")
    print("layer\thot\tcold\ttotal")
    for layer_idx in sorted(layers.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        cls = layers[layer_idx].get("classification", {})
        hot = len(cls.get("hot", []))
        cold = len(cls.get("cold", []))
        total = hot + cold
        total_hot += hot
        total_cold += cold
        print(f"{layer_idx}\t{hot}\t{cold}\t{total}")

    print("")
    print(f"TOTAL\thot={total_hot}\tcold={total_cold}\tall={total_hot + total_cold}")


if __name__ == "__main__":
    main()
