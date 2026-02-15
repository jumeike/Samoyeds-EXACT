#!/usr/bin/env python3
"""
HF evaluation using cuSPARSELt 2:4 sparse GEMM for expert Linear layers.
Uniform baseline only (hot/cold later).
"""
import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import argparse
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset

from exact_core.samoyed_converter import _prune_24_magnitude, _prune_14_magnitude, _prune_with_modelopt_24
from cusparselt_ext import CusparseLtLinear
from exact_calibration import collect_calibration_activations, prune_expert_weights, get_moe_layers


def expand_14_to_24(weight: torch.Tensor) -> torch.Tensor:
    """Expand 1:4 pruned weights to a valid 2:4 pattern by duplicating the nonzero."""
    w = weight.clone()
    nrows, ncols = w.shape
    if ncols % 4 != 0:
        raise ValueError(f"in_features must be divisible by 4, got {ncols}")
    grouped = w.view(nrows, ncols // 4, 4)
    abs_vals = grouped.abs()
    top1_idx = abs_vals.argmax(dim=2, keepdim=True)  # [rows, groups, 1]
    dup_idx = (top1_idx + 1) % 4
    # Gather top1 values and scatter into duplicate position
    top1_vals = grouped.gather(2, top1_idx)
    grouped.scatter_(2, dup_idx, top1_vals)
    return grouped.view(nrows, ncols)


def apply_uniform_cusparselt(model):
    """Replace expert Linear layers with cuSPARSELt 2:4 sparse Linear."""
    print("Applying uniform 2:4 pruning + cuSPARSELt Linear...")
    for layer in model.model.layers:
        if not hasattr(layer.mlp, 'experts'):
            continue
        for expert in layer.mlp.experts:
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                linear = getattr(expert, name)
                # Prune on CPU to reduce GPU memory pressure
                pruned_cpu = _prune_24_magnitude(linear.weight.data, return_to_device=False)
                pruned = pruned_cpu.to(linear.weight.device, dtype=linear.weight.dtype)
                new_linear = CusparseLtLinear(pruned)
                new_linear.weight.data.copy_(pruned)
                setattr(expert, name, new_linear)
    return model


def check_24_pattern(weight: torch.Tensor) -> bool:
    """Return True if weight satisfies 2:4 per row."""
    w = weight.detach().float().cpu()
    nrows, ncols = w.shape
    if ncols % 4 != 0:
        return False
    groups = w.view(nrows, ncols // 4, 4)
    nnz = (groups != 0).sum(dim=2)
    return bool((nnz == 2).all().item())


def check_14_pattern(weight: torch.Tensor) -> bool:
    """Return True if weight satisfies 1:4 per row."""
    w = weight.detach().float().cpu()
    nrows, ncols = w.shape
    if ncols % 4 != 0:
        return False
    groups = w.view(nrows, ncols // 4, 4)
    nnz = (groups != 0).sum(dim=2)
    return bool((nnz == 1).all().item())


def pack_existing_cusparselt(model, expand_14=False):
    """Wrap expert Linear layers with cuSPARSELt using existing weights."""
    print("Packing existing weights into cuSPARSELt Linear...")
    for layer in model.model.layers:
        if not hasattr(layer.mlp, 'experts'):
            continue
        for expert in layer.mlp.experts:
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                linear = getattr(expert, name)
                w = linear.weight.data.detach().cpu()
                if expand_14 and check_14_pattern(w):
                    w = expand_14_to_24(w)
                w = w.to(linear.weight.device, dtype=linear.weight.dtype)
                new_linear = CusparseLtLinear(w)
                new_linear.weight.data.copy_(w)
                setattr(expert, name, new_linear)
    return model

def apply_exact_cusparselt(model, profiling_json, cold_as_24=True, calib_acts=None, use_modelopt_mag=False, use_modelopt_sgpt=False):
    """Apply EXACT pruning but execute with cuSPARSELt (2:4 only)."""
    if not cold_as_24:
        print("Cold experts will be expanded from 1:4 to 2:4 for cuSPARSELt.")
    print(f"Applying EXACT pruning + cuSPARSELt from {profiling_json}...")
    with open(profiling_json) as f:
        prof = json.load(f)
    if 'layers' not in prof:
        raise ValueError(f"Expected 'layers' key. Got: {list(prof.keys())}")
    layers_data = prof['layers']
    for l_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, 'experts'):
            continue
        layer_key = str(l_idx)
        if layer_key not in layers_data:
            continue
        classification = layers_data[layer_key].get('classification', {})
        hot_experts = set(classification.get('hot', []))
        cold_experts = set(classification.get('cold', []))
        for e_idx, expert in enumerate(layer.mlp.experts):
            # Calibrate once per expert (not per projection)
            if calib_acts is not None:
                act = calib_acts.get((l_idx, e_idx))
                if act is None:
                    act = torch.ones(expert.gate_proj.weight.shape[1])
                if e_idx in hot_experts:
                    prune_expert_weights(expert, act, n=2, m=4)
                elif e_idx in cold_experts:
                    prune_expert_weights(expert, act, n=1, m=4)
                else:
                    prune_expert_weights(expert, act, n=2, m=4)
            # For each projection, pack into cuSPARSELt
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                linear = getattr(expert, name)
                if calib_acts is not None:
                    pruned_cpu = linear.weight.data.detach().cpu()
                else:
                    # Magnitude / ModelOpt modes
                    if e_idx in hot_experts:
                        if use_modelopt_sgpt:
                            pruned_cpu = _prune_with_modelopt_24(linear.weight.data, calib_data=None)
                        elif use_modelopt_mag:
                            pruned_cpu = _prune_with_modelopt_24(linear.weight.data)
                        else:
                            pruned_cpu = _prune_24_magnitude(linear.weight.data, return_to_device=False)
                    elif e_idx in cold_experts:
                        if cold_as_24:
                            if use_modelopt_sgpt:
                                pruned_cpu = _prune_with_modelopt_24(linear.weight.data, calib_data=None)
                            elif use_modelopt_mag:
                                pruned_cpu = _prune_with_modelopt_24(linear.weight.data)
                            else:
                                pruned_cpu = _prune_24_magnitude(linear.weight.data, return_to_device=False)
                        else:
                            pruned_cpu = _prune_14_magnitude(linear.weight.data, return_to_device=False)
                    else:
                        pruned_cpu = _prune_24_magnitude(linear.weight.data, return_to_device=False)

                if (e_idx in cold_experts) and (not cold_as_24):
                    pruned_cpu = expand_14_to_24(pruned_cpu)

                pruned = pruned_cpu.to(linear.weight.device, dtype=linear.weight.dtype)
                new_linear = CusparseLtLinear(pruned)
                new_linear.weight.data.copy_(pruned)
                setattr(expert, name, new_linear)
    return model


def get_acc(r):
    for k in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc']:
        if k in r:
            return r[k]
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen1.5-MoE-A2.7B')
    parser.add_argument('--method', choices=['baseline', 'uniform_cusparselt', 'exact_cusparselt'], required=True)
    parser.add_argument('--profiling_json', default='artifacts/exact_profiling/qwen1.5-moe-a2.7b_real_e60_layer.json')
    parser.add_argument('--cold_as_24', action='store_true', help='Force cold experts to 2:4 pruning')
    parser.add_argument('--parity_check', action='store_true', help='Run a small dense vs cuSPARSELt parity check')
    parser.add_argument('--parity_only', action='store_true', help='Run parity check and exit')
    parser.add_argument('--validate_24', action='store_true', help='Check 2:4 pattern on first expert weight')
    parser.add_argument('--use_calibration', action='store_true', help='Use calibration (SparseGPT-like) pruning')
    parser.add_argument('--use_modelopt_mag', action='store_true', help='Use ModelOpt magnitude pruning for 2:4')
    parser.add_argument('--use_modelopt_sgpt', action='store_true', help='Use ModelOpt SparseGPT for 2:4 (uses calib)')
    parser.add_argument('--calib_dataset', default='c4', choices=['wikitext', 'c4', 'eval_tasks'])
    parser.add_argument('--calib_samples', type=int, default=512)
    parser.add_argument('--calib_max_length', type=int, default=512)
    parser.add_argument('--calib_eval_tasks', default='arc_challenge,winogrande,mmlu_astronomy,openbookqa,gsm8k')
    parser.add_argument('--tasks', default='mmlu_astronomy')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pack_only', action='store_true',
                        help='Do not re-prune; just pack existing weights into cuSPARSELt.')
    parser.add_argument('--expand_14_to_24', action='store_true',
                        help='When packing, expand any 1:4 weights to valid 2:4.')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("=" * 70)
    print(f"METHOD: {args.method.upper()}")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    print("✓ Loaded")

    calib_acts = None
    if (args.use_calibration or args.use_modelopt_sgpt) and args.method == 'exact_cusparselt':
        print(f"Collecting calibration activations from {args.calib_dataset} ({args.calib_samples} samples)...")
        calib_texts = []
        if args.calib_dataset == 'wikitext':
            ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            for row in ds:
                if row.get('text', '').strip():
                    calib_texts.append(row['text'])
                if len(calib_texts) >= args.calib_samples:
                    break
        elif args.calib_dataset == 'c4':
            ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
            for row in ds:
                if row.get('text', '').strip():
                    calib_texts.append(row['text'])
                if len(calib_texts) >= args.calib_samples:
                    break
        else:
            tasks = [t.strip() for t in args.calib_eval_tasks.split(',') if t.strip()]
            for t in tasks:
                if t == 'arc_challenge':
                    ds = load_dataset('ai2_arc', 'ARC-Challenge', split='test')
                    for row in ds:
                        q = row.get('question', '')
                        choices = row.get('choices', {})
                        if isinstance(choices, dict) and 'text' in choices and 'label' in choices:
                            c = "\n".join([f"{l}. {tx}" for l, tx in zip(choices['label'], choices['text'])])
                        else:
                            c = ""
                        calib_texts.append(f"{q}\n{c}")
                        if len(calib_texts) >= args.calib_samples:
                            break
                if len(calib_texts) >= args.calib_samples:
                    break
        moe_layers = get_moe_layers(model)
        calib_acts = collect_calibration_activations(model, tokenizer, calib_texts, moe_layers)

    if args.pack_only and args.method in ['uniform_cusparselt', 'exact_cusparselt']:
        model = pack_existing_cusparselt(model, expand_14=args.expand_14_to_24)
    elif args.method == 'uniform_cusparselt':
        model = apply_uniform_cusparselt(model)
    elif args.method == 'exact_cusparselt':
        model = apply_exact_cusparselt(
            model,
            args.profiling_json,
            cold_as_24=args.cold_as_24,
            calib_acts=calib_acts if args.use_calibration else None,
            use_modelopt_mag=args.use_modelopt_mag,
            use_modelopt_sgpt=args.use_modelopt_sgpt
        )

    model.eval()

    if args.parity_check or args.validate_24:
        # pick first expert
        first_layer = None
        for layer in model.model.layers:
            if hasattr(layer.mlp, 'experts'):
                first_layer = layer
                break
        if first_layer is not None:
            expert0 = first_layer.mlp.experts[0]
            # gate_proj should be CusparseLtLinear if using cusparselt
            if hasattr(expert0, 'gate_proj') and hasattr(expert0.gate_proj, 'weight'):
                w = expert0.gate_proj.weight.data
                if args.validate_24:
                    ok = check_24_pattern(w)
                    print(f"2:4 pattern check (expert0 gate_proj): {ok}")
                if args.parity_check:
                    x = torch.randn(16, w.shape[1], device=w.device, dtype=w.dtype)
                    y_sparse = expert0.gate_proj(x)
                    y_dense = x.matmul(w.t())
                    diff = (y_sparse - y_dense).abs()
                    print(f"Parity check gate_proj: max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
        if args.parity_only:
            return

    print(f"\nEvaluating: {args.tasks}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    t0 = time.time()
    results = evaluator.simple_evaluate(model=lm, tasks=args.tasks.split(','), batch_size=args.batch_size)
    t1 = time.time()
    print(f"\nTotal eval time: {t1 - t0:.1f}s")

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Task':<20} {'Accuracy'}")
    print("-" * 35)
    for task in args.tasks.split(','):
        print(f"{task:<20} {get_acc(results['results'][task]) * 100:>6.2f}%")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"MAX_MEMORY_ALLOCATED_GB: {peak_gb:.3f}")


if __name__ == "__main__":
    main()
