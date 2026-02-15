#!/usr/bin/env python3
"""
HF-only evaluation for pruning accuracy (no Samoyeds kernels).
Mixtral dense compute.
Methods:
  - baseline: dense HF model
  - uniform_dense: uniform 2:4 pruning on experts
  - exact_dense: adaptive hot/cold pruning from profiling JSON
"""
import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from exact_core.samoyed_converter import _prune_24_magnitude, _prune_14_magnitude


def _get_experts_from_layer(layer):
    if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
        return layer.block_sparse_moe.experts
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
        return layer.mlp.experts
    return None


def apply_uniform_dense(model):
    """Apply uniform 2:4 pruning to experts in-place (HF dense compute)."""
    print("Applying uniform 2:4 pruning (dense compute)...")
    for layer in model.model.layers:
        experts = _get_experts_from_layer(layer)
        if experts is None:
            continue
        for expert in experts:
            for name in ['w1', 'w2', 'w3']:
                weight = getattr(expert, name).weight
                pruned = _prune_24_magnitude(weight.data)
                weight.data.copy_(pruned.to(weight.device, dtype=weight.dtype))
    return model


def apply_exact_dense(model, profiling_json, cold_as_24=False):
    """Apply EXACT hot/cold pruning to experts in-place (HF dense compute)."""
    print(f"Applying EXACT pruning (dense compute) from {profiling_json}...")
    with open(profiling_json) as f:
        prof = json.load(f)
    if 'layers' not in prof:
        raise ValueError(f"Expected 'layers' key. Got: {list(prof.keys())}")
    layers_data = prof['layers']

    for l_idx, layer in enumerate(model.model.layers):
        experts = _get_experts_from_layer(layer)
        if experts is None:
            continue
        layer_key = str(l_idx)
        if layer_key not in layers_data:
            continue
        classification = layers_data[layer_key].get('classification', {})
        hot_experts = set(classification.get('hot', []))
        cold_experts = set(classification.get('cold', []))

        for e_idx, expert in enumerate(experts):
            if e_idx in hot_experts:
                pruner = _prune_24_magnitude
            elif e_idx in cold_experts:
                pruner = _prune_24_magnitude if cold_as_24 else _prune_14_magnitude
            else:
                pruner = _prune_24_magnitude
            for name in ['w1', 'w2', 'w3']:
                weight = getattr(expert, name).weight
                pruned = pruner(weight.data)
                weight.data.copy_(pruned.to(weight.device, dtype=weight.dtype))
    return model


def get_acc(r):
    for k in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc']:
        if k in r:
            return r[k]
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mistralai/Mixtral-8x7B-v0.1')
    parser.add_argument('--method', choices=['baseline', 'uniform_dense', 'exact_dense'], required=True)
    parser.add_argument('--profiling_json', default='artifacts/exact_profiling/mixtral-8x7b-v0.1_c4_real_e8_layer.json')
    parser.add_argument('--tasks', default='mmlu_astronomy')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cold_as_24', action='store_true', help='Force cold experts to 2:4 pruning')
    parser.add_argument('--device_map', default='auto',
                        help="Device map for model loading (e.g., 'auto', 'cpu', or 'none')")
    parser.add_argument('--low_cpu_mem_usage', action=argparse.BooleanOptionalAction, default=True,
                        help='Use low_cpu_mem_usage when loading (set to False to avoid meta tensors)')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16',
                        help='Model dtype for loading')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("=" * 70)
    print(f"METHOD: {args.method.upper()}")
    print("=" * 70)

    device_map = None if args.device_map == 'none' else args.device_map
    if (device_map is not None) and (not args.low_cpu_mem_usage):
        print("WARNING: device_map requires low_cpu_mem_usage=True; disabling device_map.")
        device_map = None
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    if device_map is None and torch_dtype == torch.float16 and not torch.cuda.is_available():
        print("WARNING: float16 on CPU is unsupported; switching to float32.")
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=True
    )
    print("✓ Loaded")

    if args.method == 'uniform_dense':
        model = apply_uniform_dense(model)
    elif args.method == 'exact_dense':
        model = apply_exact_dense(model, args.profiling_json, cold_as_24=args.cold_as_24)

    model.eval()

    print(f"\nEvaluating: {args.tasks}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(model=lm, tasks=args.tasks.split(','), batch_size=args.batch_size)

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
