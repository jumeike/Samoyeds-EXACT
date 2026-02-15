#!/usr/bin/env python3
"""
Sweep fixed hot ratios by generating profiles and invoking eval_qwen_cusparselt.py.
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


def make_fixed_ratio_profile(input_json, hot_ratio, metric="activation_rate"):
    with open(input_json, "r") as f:
        prof = json.load(f)
    layers = prof.get("layers", {})
    for _, layer_data in layers.items():
        stats = layer_data.get("expert_stats", {})
        if not stats:
            continue
        items = []
        for k, v in stats.items():
            items.append((int(k), float(v.get(metric, 0.0))))
        items.sort(key=lambda x: x[1], reverse=True)
        num_experts = len(items)
        k_hot = max(1, int(round(num_experts * hot_ratio)))
        hot = sorted([eid for eid, _ in items[:k_hot]])
        cold = sorted([eid for eid, _ in items[k_hot:]])
        layer_data["classification"] = {"hot": hot, "cold": cold}
    out_dir = os.path.join(os.path.dirname(input_json), "fixed_ratio")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_json))[0]
    out_path = os.path.join(out_dir, f"{stem}_ratio{hot_ratio:.3f}.json")
    with open(out_path, "w") as f:
        json.dump(prof, f, indent=2)
    return out_path


def profile_stats(profile_path):
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
    eff_density_24 = hot_ratio * 0.5 + cold_ratio * 0.5
    mem_saved_24 = 1.0 - eff_density_24
    eff_density_14 = hot_ratio * 0.5 + cold_ratio * 0.25
    mem_saved_14 = 1.0 - eff_density_14
    return {
        "hot_ratio": hot_ratio,
        "cold_ratio": cold_ratio,
        "eff_density_hot24_cold24": eff_density_24,
        "mem_saved_hot24_cold24": mem_saved_24,
        "eff_density_hot24_cold14": eff_density_14,
        "mem_saved_hot24_cold14": mem_saved_14,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--ratios", default="0.1,0.2,0.25,0.3,0.4")
    parser.add_argument("--metric", choices=["activation_rate", "count"], default="activation_rate")
    parser.add_argument("--tasks", default="mmlu_astronomy")
    parser.add_argument("--method", default="exact_cusparselt")
    parser.add_argument("--cold_as_24", action="store_true")
    parser.add_argument("--output_csv", default="artifacts/results/hot_ratio_sweep.csv")
    args = parser.parse_args()

    ratios = [r.strip() for r in args.ratios.split(",") if r.strip()]
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

    for r in ratios:
        hot_ratio = float(r)
        prof = make_fixed_ratio_profile(args.input_json, hot_ratio, metric=args.metric)
        stats = profile_stats(prof)

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

        print(f"[sweep] hot_ratio={hot_ratio:.3f} cold_as_24 cmd={' '.join(cmd_cold24)}", flush=True)
        ok24, acc24, mem24_gb, res24 = run_eval(cmd_cold24, tasks)
        if res24.stdout:
            print(res24.stdout, flush=True)
        if res24.stderr:
            print(res24.stderr, flush=True)

        print(f"[sweep] hot_ratio={hot_ratio:.3f} true_adaptive cmd={' '.join(cmd_true)}", flush=True)
        ok14, acc14, mem14_gb, res14 = run_eval(cmd_true, tasks)
        if res14.stdout:
            print(res14.stdout, flush=True)
        if res14.stderr:
            print(res14.stderr, flush=True)

        status = "ok" if ok24 and ok14 and base_ok else "failed"
        row = {
            "hot_ratio_target": f"{hot_ratio:.3f}",
            "status": status,
            "hot_ratio_actual": f"{stats['hot_ratio']:.4f}",
            "cold_ratio_actual": f"{stats['cold_ratio']:.4f}",
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

        if not ok24:
            rows[-1]["stderr_tail_cold24"] = res24.stderr[-400:]
        if not ok14:
            rows[-1]["stderr_tail_cold14"] = res14.stderr[-400:]
        if not base_ok:
            rows[-1]["stderr_tail_dense"] = base_res.stderr[-400:]

        if out_fields is None:
            out_fields = sorted(row.keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            for rr in rows:
                writer.writerow(rr)

    print(f"Saved sweep results to {output_csv} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
