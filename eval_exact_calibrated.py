#!/usr/bin/env python3
import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset
import argparse
import csv
from datetime import datetime
import gc

from exact_calibration import (
    collect_calibration_activations,
    apply_calibrated_pruning,
    fine_tune_pruned,
    get_moe_layers
)

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--profiling_json', required=True)
parser.add_argument('--tasks', default='mmlu,hellaswag,arc_challenge,winogrande')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--calib_samples', type=int, default=512)
parser.add_argument('--finetune_steps', type=int, default=500)
parser.add_argument('--skip_baseline', action='store_true')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

# Baseline evaluation
baseline_results = None
if not args.skip_baseline:
    print("="*70)
    print("BASELINE EVALUATION")
    print("="*70)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True
    )
    baseline_model.eval()
    
    baseline_lm = HFLM(pretrained=baseline_model, tokenizer=tokenizer)
    baseline_results = evaluator.simple_evaluate(
        model=baseline_lm, tasks=args.tasks.split(','), batch_size=args.batch_size
    )
    
    del baseline_model, baseline_lm
    gc.collect()
    torch.cuda.empty_cache()

# EXACT with calibration
print("\n" + "="*70)
print("EXACT WITH CALIBRATION")
print("="*70)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True
)

# Get calibration data
print(f"Loading {args.calib_samples} calibration samples...")
calib_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
calib_texts = [s['text'] for s in calib_dataset if s['text'].strip()][:args.calib_samples]

# Collect activations
# model = model.cuda()
model.eval()
moe_layers = get_moe_layers(model)
calib_activations = collect_calibration_activations(
    model, tokenizer, calib_texts, moe_layers, #device='cuda'
)

# Apply calibrated pruning (on CPU to save memory)
# model = model.cpu()
# torch.cuda.empty_cache()
model = apply_calibrated_pruning(model, args.profiling_json, calib_activations)

# Fine-tune
if args.finetune_steps > 0:
    # model = model.cuda()
    model.gradient_checkpointing_enable()
    fine_tune_pruned(model, tokenizer, calib_texts, steps=args.finetune_steps)
    # model = model.cpu()
    # torch.cuda.empty_cache()

# Evaluate
# model = model.cuda()
model.eval()
exact_lm = HFLM(pretrained=model, tokenizer=tokenizer)
exact_results = evaluator.simple_evaluate(
    model=exact_lm, tasks=args.tasks.split(','), batch_size=args.batch_size
)

# Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

def get_metric(task_result):
    for key in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc']:
        if key in task_result:
            return task_result[key]
    return next((v for k, v in task_result.items() 
                if isinstance(v, (int, float)) and k != 'alias'), 0)

results_data = []
print(f"\n{'Task':<20} {'Baseline':<12} {'EXACT':<12} {'Drop':<12}")
print("-" * 70)

for task in args.tasks.split(','):
    if baseline_results:
        baseline_acc = get_metric(baseline_results['results'][task])
        exact_acc = get_metric(exact_results['results'][task])
        drop = (baseline_acc - exact_acc) * 100
        
        print(f"{task:<20} {baseline_acc*100:>6.2f}%      {exact_acc*100:>6.2f}%      {drop:>+6.2f}%")
        results_data.append({
            'task': task, 'baseline_acc': baseline_acc * 100,
            'exact_acc': exact_acc * 100, 'accuracy_drop': drop
        })
    else:
        exact_acc = get_metric(exact_results['results'][task])
        print(f"{task:<20} {'SKIPPED':<12} {exact_acc*100:>6.2f}%      {'N/A':<12}")
        results_data.append({
            'task': task, 'baseline_acc': 'N/A',
            'exact_acc': exact_acc * 100, 'accuracy_drop': 'N/A'
        })

# Save
if results_data:
    model_name = args.model.replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = 'artifacts/results/calibrated_exact'
    os.makedirs(results_dir, exist_ok=True)
    
    csv_file = os.path.join(results_dir, f"exact_{model_name}_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task', 'baseline_acc', 'exact_acc', 'accuracy_drop'])
        writer.writeheader()
        writer.writerows(results_data)
    
    print(f"\nSaved: {csv_file}")
