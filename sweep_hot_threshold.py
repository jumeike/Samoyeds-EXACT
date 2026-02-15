#!/usr/bin/env python3
"""
Run a hot-threshold sweep by invoking eval_qwen_cusparselt.py and collecting accuracy.
"""
import argparse
import os
import re
import subprocess
import csv
import json
from datetime import datetime


def parse_accuracy(stdout, task):
    pattern = re.compile(rf"^\\s*{re.escape(task)}\\s+(\\d+\\.\\d+)%", re.MULTILINE)
    m = pattern.search(stdout)
    return float(m.group(1)) if m else None


def parse_memory(stdout):
    m = re.search(r"MAX_MEMORY_ALLOCATED_GB:\\s*(\\d+\\.\\d+)", stdout)
    return float(m.group(1)) if m else None


def run_eval(cmd, tasks):
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout.replace("\r", "\n")
    # Strip ANSI escape codes if present
    out = re.sub(r"\x1b\\[[0-9;]*m", "", out)
    acc_map = {task: parse_accuracy(out, task) for task in tasks}
    mem_gb = parse_memory(out)
    ok = result.returncode == 0 and all(v is not None for v in acc_map.values())
    return ok, acc_map, mem_gb, result


def profile_stats(profile_path, cold_as_24=False):
    with open(profile_path, "r") as f:
        prof = json.load(f)
    layers = prof.get("layers", {})
    total_experts = 0
    hot_experts = 0
    cold_experts = 0
    for _, layer_data in layers.items():
        cls = layer_data.get("classification", {})
        hot = cls.get("hot", [])
        cold = cls.get("cold", [])
        hot_experts += len(hot)
        cold_experts += len(cold)
        total_experts += len(hot) + len(cold)
    hot_ratio = hot_experts / total_experts if total_experts else 0.0
    cold_ratio = cold_experts / total_experts if total_experts else 0.0
    # Effective density: 2:4 -> 0.5, 1:4 -> 0.25
    cold_density = 0.5 if cold_as_24 else 0.25
    eff_density = hot_ratio * 0.5 + cold_ratio * cold_density
    mem_saved = 1.0 - eff_density
    eff_density_24 = hot_ratio * 0.5 + cold_ratio * 0.5
    mem_saved_24 = 1.0 - eff_density_24
    eff_density_14 = hot_ratio * 0.5 + cold_ratio * 0.25
    mem_saved_14 = 1.0 - eff_density_14
    return {
        "hot_ratio": hot_ratio,
        "cold_ratio": cold_ratio,
        "eff_density": eff_density,
        "mem_saved": mem_saved,
        "eff_density_hot24_cold24": eff_density_24,
        "mem_saved_hot24_cold24": mem_saved_24,
        "eff_density_hot24_cold14": eff_density_14,
        "mem_saved_hot24_cold14": mem_saved_14,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--tasks", default="mmlu_astronomy")
    parser.add_argument("--thresholds", default="0.005,0.007,0.01,0.015,0.02,0.05")
    parser.add_argument("--profiling_dir", default="artifacts/exact_profiling")
    parser.add_argument("--method", default="exact_cusparselt")
    parser.add_argument("--cold_as_24", action="store_true")
    parser.add_argument("--output_csv", default="artifacts/results/hot_threshold_sweep.csv")
    args = parser.parse_args()

    thresholds = [t.strip() for t in args.thresholds.split(",") if t.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root, ext = os.path.splitext(args.output_csv)
    output_csv = f"{root}_{stamp}{ext or '.csv'}"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    rows = []
    out_fields = None

    base_cmd = [
        "python",
        "eval_qwen_dense.py",
        "--method",
        "baseline",
        "--tasks",
        args.tasks,
    ]
    print(f"[sweep] baseline cmd={' '.join(base_cmd)}", flush=True)
    base_ok, base_acc_map, base_mem_gb, base_res = run_eval(base_cmd, tasks)
    if base_res.stdout:
        print(base_res.stdout, flush=True)
    if base_res.stderr:
        print(base_res.stderr, flush=True)

    for t in thresholds:
        prof = os.path.join(
            args.profiling_dir,
            f"qwen1.5-moe-a2.7b_c4_ht{t}_real_e60_layer.json",
        )
        if not os.path.exists(prof):
            rows.append({"threshold": t, "status": "missing_profile"})
            continue
        stats = profile_stats(prof, cold_as_24=args.cold_as_24)

        cmd_true = [
            "python",
            "eval_qwen_cusparselt.py",
            "--method",
            args.method,
            "--profiling_json",
            prof,
            "--tasks",
            args.tasks,
        ]
        cmd_cold24 = cmd_true.copy()
        cmd_cold24.append("--cold_as_24")

        print(f"[sweep] threshold={t} cold_as_24 cmd={' '.join(cmd_cold24)}", flush=True)
        ok24, acc24, mem24_gb, res24 = run_eval(cmd_cold24, tasks)
        if res24.stdout:
            print(res24.stdout, flush=True)
        if res24.stderr:
            print(res24.stderr, flush=True)

        print(f"[sweep] threshold={t} true_adaptive cmd={' '.join(cmd_true)}", flush=True)
        ok14, acc14, mem14_gb, res14 = run_eval(cmd_true, tasks)
        if res14.stdout:
            print(res14.stdout, flush=True)
        if res14.stderr:
            print(res14.stderr, flush=True)

        status = "ok" if ok24 and ok14 and base_ok else "failed"
        row = {
            "threshold": t,
            "status": status,
            "hot_ratio": f"{stats['hot_ratio']:.4f}",
            "cold_ratio": f"{stats['cold_ratio']:.4f}",
            "eff_density": f"{stats['eff_density']:.4f}",
            "mem_saved": f"{stats['mem_saved']:.4f}",
            "eff_density_hot24_cold24": f"{stats['eff_density_hot24_cold24']:.4f}",
            "mem_saved_hot24_cold24": f"{stats['mem_saved_hot24_cold24']:.4f}",
            "eff_density_hot24_cold14": f"{stats['eff_density_hot24_cold14']:.4f}",
            "mem_saved_hot24_cold14": f"{stats['mem_saved_hot24_cold14']:.4f}",
        }
        for task in tasks:
            row[f"acc_cold24_{task}"] = acc24.get(task) if acc24.get(task) is not None else ""
            row[f"acc_cold14_{task}"] = acc14.get(task) if acc14.get(task) is not None else ""
            row[f"acc_dense_{task}"] = base_acc_map.get(task) if base_acc_map.get(task) is not None else ""
        row["mem_gb_cold24"] = f"{mem24_gb:.3f}" if mem24_gb is not None else ""
        row["mem_gb_cold14"] = f"{mem14_gb:.3f}" if mem14_gb is not None else ""
        row["mem_gb_dense"] = f"{base_mem_gb:.3f}" if base_mem_gb is not None else ""
        rows.append(row)
        # Write incremental results after each run
        if out_fields is None:
            out_fields = sorted(row.keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        if not ok24:
            rows[-1]["stderr_tail_cold24"] = res24.stderr[-400:]
        if not ok14:
            rows[-1]["stderr_tail_cold14"] = res14.stderr[-400:]
        if not base_ok:
            rows[-1]["stderr_tail_dense"] = base_res.stderr[-400:]

    print(f"Saved sweep results to {output_csv} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
