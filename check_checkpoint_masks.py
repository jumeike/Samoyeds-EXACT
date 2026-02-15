#!/usr/bin/env python3
import argparse
import json

import torch
from transformers import AutoModelForCausalLM


def avg_nonzeros_per_group(weight: torch.Tensor, group_size: int = 4) -> float:
    x = weight.detach().float().cpu()
    in_dim = x.shape[1]
    usable = (in_dim // group_size) * group_size
    if usable == 0:
        return 0.0
    x = x[:, :usable].reshape(x.shape[0], -1, group_size)
    nz = (x != 0).sum(dim=-1).float()
    return float(nz.mean().item())


def main():
    parser = argparse.ArgumentParser(description="Validate hot/cold N:M masks in a finetuned checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path or HF name of checkpoint to inspect.")
    parser.add_argument("--profiling_json", required=True, help="Profiling JSON with hot/cold classification.")
    parser.add_argument("--mode", choices=["exact", "uniform"], default="exact",
                        help="Mask target mode. exact=hot 2:4, cold 1:4 (unless --cold_as_24). uniform=all 2:4.")
    parser.add_argument("--cold_as_24", action="store_true", help="Expect cold experts to be 2:4.")
    parser.add_argument("--tol", type=float, default=0.05, help="Tolerance for avg nonzeros per 4-group.")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    with open(args.profiling_json, "r") as f:
        prof = json.load(f)

    hot_ok = cold_ok = hot_tot = cold_tot = 0
    hot_ratios = []
    cold_ratios = []

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("Unsupported model structure: expected model.model.layers")

    for l_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "experts"):
            continue
        cls = prof.get("layers", {}).get(str(l_idx), {}).get("classification", {})
        hot = set(cls.get("hot", []))
        cold = set(cls.get("cold", []))
        experts = layer.mlp.experts

        for e_idx, expert in enumerate(experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                if not hasattr(expert, proj):
                    continue
                ratio = avg_nonzeros_per_group(getattr(expert, proj).weight, group_size=4)

                if e_idx in hot:
                    hot_tot += 1
                    hot_ratios.append(ratio)
                    if abs(ratio - 2.0) <= args.tol:
                        hot_ok += 1
                elif e_idx in cold:
                    cold_tot += 1
                    cold_ratios.append(ratio)
                    target = 2.0 if (args.mode == "uniform" or args.cold_as_24) else 1.0
                    if abs(ratio - target) <= args.tol:
                        cold_ok += 1

    hot_avg = (sum(hot_ratios) / len(hot_ratios)) if hot_ratios else 0.0
    cold_avg = (sum(cold_ratios) / len(cold_ratios)) if cold_ratios else 0.0

    print(f"HOT  avg_nonzeros_per_4={hot_avg:.4f} pass={hot_ok}/{hot_tot}")
    print(f"COLD avg_nonzeros_per_4={cold_avg:.4f} pass={cold_ok}/{cold_tot}")

    hot_pass_rate = (hot_ok / hot_tot) if hot_tot else 0.0
    cold_pass_rate = (cold_ok / cold_tot) if cold_tot else 0.0
    if hot_pass_rate < 0.99 or cold_pass_rate < 0.99:
        print("WARNING: Mask pattern mismatch detected.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
