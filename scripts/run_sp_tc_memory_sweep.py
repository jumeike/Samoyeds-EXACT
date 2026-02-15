#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


RE_FLOAT_GB = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*GB")
RE_FLOAT_X = re.compile(r"([0-9]+(?:\.[0-9]+)?)x")
RE_FLOAT_PCT = re.compile(r"([0-9]+(?:\.[0-9]+)?)%")


def parse_sp_tc_output(text: str):
    dense_gb = compressed_gb = saved_gb = None
    ratio_x = compressed_pct = saved_pct = None

    for line in text.splitlines():
        if line.startswith("Dense bytes:"):
            m = RE_FLOAT_GB.search(line)
            dense_gb = float(m.group(1)) if m else None
        elif line.startswith("Compressed bytes:"):
            m = RE_FLOAT_GB.search(line)
            compressed_gb = float(m.group(1)) if m else None
        elif line.startswith("Compression ratio (dense/compressed):"):
            m = RE_FLOAT_X.search(line)
            ratio_x = float(m.group(1)) if m else None
        elif line.startswith("Compressed fraction (compressed/dense):"):
            m = RE_FLOAT_PCT.search(line)
            compressed_pct = float(m.group(1)) if m else None
        elif line.startswith("Saved:"):
            gb = RE_FLOAT_GB.search(line)
            pct = RE_FLOAT_PCT.search(line)
            saved_gb = float(gb.group(1)) if gb else None
            saved_pct = float(pct.group(1)) if pct else None

    if dense_gb is None or compressed_gb is None:
        raise RuntimeError("Failed to parse sp_tc_memory output:\n" + text)

    return {
        "dense_gb": dense_gb,
        "compressed_gb": compressed_gb,
        "compression_ratio_x": ratio_x,
        "compressed_fraction_pct": compressed_pct,
        "saved_gb": saved_gb,
        "saved_pct": saved_pct,
    }


def run_sp_tc(script_path: Path, model_name: str, profiling_json: str):
    cmd = [
        "python",
        str(script_path),
        "--model",
        model_name,
        "--profiling_json",
        profiling_json,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    parsed = parse_sp_tc_output(proc.stdout)
    return parsed, proc.stdout


def make_all_hot_profile(src_json: Path, dst_json: Path):
    with src_json.open("r") as f:
        data = json.load(f)

    layers = data.get("layers", {})
    for _, layer_data in layers.items():
        cls = layer_data.get("classification", {})
        hot = cls.get("hot", [])
        cold = cls.get("cold", [])
        n = len(hot) + len(cold)
        cls["hot"] = list(range(n))
        cls["cold"] = []
        layer_data["classification"] = cls

    with dst_json.open("w") as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Run sp_tc_memory.py across models/policies and save CSV."
    )
    parser.add_argument(
        "--repo_root",
        default=".",
        help="Path to Samoyeds-EXACT repo root (default: current directory).",
    )
    parser.add_argument(
        "--output_csv",
        default="artifacts/results/memory/sp_tc_memory_sweep.csv",
        help="Output CSV path (repo-relative by default).",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    sp_tc = repo / "sp_tc_memory.py"
    out_csv = (repo / args.output_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    models = [
        {
            "key": "deepseek",
            "hf_model": "deepseek-ai/deepseek-moe-16b-base",
            "profiles": {
                "mass0.7": repo / "artifacts/exact_profiling/deepseek-moe-16b-base/deepseek-moe-16b-base_eval_tasks_mass0.7_real_e64_layer.json",
                "mass0.8": repo / "artifacts/exact_profiling/deepseek-moe-16b-base/deepseek-moe-16b-base_eval_tasks_mass0.8_real_e64_layer.json",
                "mass0.9": repo / "artifacts/exact_profiling/deepseek-moe-16b-base/deepseek-moe-16b-base_eval_tasks_mass0.9_real_e64_layer.json",
            },
        },
        {
            "key": "qwen",
            "hf_model": "Qwen/Qwen1.5-MoE-A2.7B",
            "profiles": {
                "mass0.7": repo / "artifacts/exact_profiling/qwen1.5-moe-a2.7b/qwen1.5-moe-a2.7b_eval_tasks_mass0.7_real_e60_layer.json",
                "mass0.8": repo / "artifacts/exact_profiling/qwen1.5-moe-a2.7b/qwen1.5-moe-a2.7b_eval_tasks_mass0.8_real_e60_layer.json",
                "mass0.9": repo / "artifacts/exact_profiling/qwen1.5-moe-a2.7b/qwen1.5-moe-a2.7b_eval_tasks_mass0.9_real_e60_layer.json",
            },
        },
        {
            "key": "mixtral",
            "hf_model": "mistralai/Mixtral-8x7B-v0.1",
            "profiles": {
                "mass0.7": repo / "artifacts/exact_profiling/mixtral-8x7b-v0.1/mixtral-8x7b-v0.1_eval_tasks_mass0.7_real_e8_layer.json",
                "mass0.8": repo / "artifacts/exact_profiling/mixtral-8x7b-v0.1/mixtral-8x7b-v0.1_eval_tasks_mass0.8_real_e8_layer.json",
                "mass0.9": repo / "artifacts/exact_profiling/mixtral-8x7b-v0.1/mixtral-8x7b-v0.1_eval_tasks_mass0.9_real_e8_layer.json",
            },
        },
    ]

    rows = []

    for m in models:
        key = m["key"]
        hf_model = m["hf_model"]
        profiles = m["profiles"]

        # mass0.7 / 0.8 / 0.9
        for policy, p in profiles.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing profile: {p}")
            parsed, _ = run_sp_tc(sp_tc, hf_model, str(p))
            rows.append(
                {
                    "model": key,
                    "policy": policy,
                    **parsed,
                }
            )

        # uniform (all-hot) profile from mass0.9
        with tempfile.TemporaryDirectory() as td:
            tmp_uniform = Path(td) / f"{key}_uniform_all_hot.json"
            make_all_hot_profile(profiles["mass0.9"], tmp_uniform)
            parsed_uniform, _ = run_sp_tc(sp_tc, hf_model, str(tmp_uniform))
            rows.append(
                {
                    "model": key,
                    "policy": "uniform_all_hot",
                    **parsed_uniform,
                }
            )

        # baseline row (dense)
        # Reuse dense_gb from mass0.9 run for this model (dense is invariant across policies).
        dense_ref = next(r for r in rows if r["model"] == key and r["policy"] == "mass0.9")
        rows.append(
            {
                "model": key,
                "policy": "baseline_dense",
                "dense_gb": dense_ref["dense_gb"],
                "compressed_gb": dense_ref["dense_gb"],
                "compression_ratio_x": 1.0,
                "compressed_fraction_pct": 100.0,
                "saved_gb": 0.0,
                "saved_pct": 0.0,
            }
        )

    fieldnames = [
        "model",
        "policy",
        "dense_gb",
        "compressed_gb",
        "compression_ratio_x",
        "compressed_fraction_pct",
        "saved_gb",
        "saved_pct",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {out_csv}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
