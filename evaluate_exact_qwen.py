import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import argparse
import csv
from datetime import datetime

from qwen2_moe.modeling_qwen2_moe_samoyeds import SSQwen2MoeForCausalLM
from qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from exact_weight_loader_qwen import load_pretrained_weights_into_samoyeds
from exact_core import apply_exact_to_model
from memory_tracker import MemoryTracker, compare_model_sizes

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Qwen/Qwen1.5-MoE-A2.7B')
parser.add_argument('--profiling_json', default='artifacts/exact_profiling/qwen1.5-moe-a2.7b_real_e60_layer.json')
parser.add_argument('--tasks', default='mmlu,hellaswag,arc_challenge,winogrande,openbookqa,gsm8k')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline evaluation')
args = parser.parse_args()

# Initialize memory tracker
memory_tracker = MemoryTracker()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

# Evaluate official pretrained model as baseline
if not args.skip_baseline:
    print("="*70)
    print("EVALUATING OFFICIAL PRETRAINED MODEL (BASELINE)")
    print("="*70)
    print(f"Loading pretrained model from {args.model}...")

    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    baseline_model.eval()
    
    with memory_tracker.track('baseline_inference'):
        print("\nEvaluating baseline model...")
        baseline_lm = HFLM(pretrained=baseline_model, tokenizer=tokenizer)
        baseline_results = evaluator.simple_evaluate(
            model=baseline_lm,
            tasks=args.tasks.split(','),
            batch_size=args.batch_size
        )
    print("✓ Baseline evaluation complete")

    # Free GPU memory
    print("Freeing GPU memory...")
    del baseline_model, baseline_lm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("✓ GPU memory cleared")
else:
    baseline_results = None

# Create sparse Samoyeds model for EXACT compression
print("\n" + "="*70)
print("EVALUATING SAMOYEDS QWEN WITH EXACT COMPRESSION")
print("="*70)

# Create dense model for baseline evaluation
config = Qwen2MoeConfig.from_pretrained(args.model, trust_remote_code=True)

# Print model configuration
print(f"Qwen1.5-MoE-A2.7B Configuration:")
print(f"  Model size: {config.num_hidden_layers} layers, {config.hidden_size} hidden dim")
print(f"  MoE: {config.num_experts} routed experts + shared expert")
print(f"  Routing: top-{config.num_experts_per_tok}, every {config.decoder_sparse_step} layer(s)")
print(f"  Attention: {config.num_attention_heads} heads (MHA)")
print(f"  FFN: shared={config.shared_expert_intermediate_size}, MoE={config.moe_intermediate_size} per expert")
print(f"  Context: {config.max_position_embeddings} tokens")

print("Creating Dense Qwen2 model...")
dense_model = SSQwen2MoeForCausalLM(config, use_dense=True).half()

print(f"Loading pretrained weights from {args.model}...")
dense_model = load_pretrained_weights_into_samoyeds(dense_model, args.model)
# dense_model = dense_model.cuda()
# dense_model.eval()
print("✓ Weights loaded")

print("Creating Samoyeds Qwen2 model...")
samoyeds_model = SSQwen2MoeForCausalLM(config, use_dense=False).half()
# samoyeds_model = samoyeds_model.cuda()
# samoyeds_model.eval()

# Apply EXACT compression (uses dense_model weights)
print(f"\nApplying EXACT compression from {args.profiling_json}...")
exact_model = apply_exact_to_model(samoyeds_model, args.profiling_json, dense_model=dense_model)
# exact_model = exact_model.cuda()
# exact_model.eval()
print("✓ EXACT compression applied")

# Free uncompressed models
del dense_model, samoyeds_model
import gc
gc.collect()

# Move compressed model to GPU
print("Moving compressed model to GPU...")
exact_model = exact_model.cuda()
exact_model.eval()
print("✓ Model loaded to GPU")

# Compare model sizes before evaluation
if not args.skip_baseline:
    # Need to reload baseline for comparison since we deleted it
    baseline_for_comparison = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True
    )
    model_comparison = compare_model_sizes(baseline_for_comparison, exact_model)
    del baseline_for_comparison
    gc.collect()
else:
    model_comparison = {}

# Evaluate EXACT model with memory tracking
print("\nEvaluating EXACT compressed model...")
with memory_tracker.track('exact_inference'):
    exact_lm = HFLM(pretrained=exact_model, tokenizer=tokenizer)
    exact_results = evaluator.simple_evaluate(
        model=exact_lm,
        tasks=args.tasks.split(','),
        batch_size=args.batch_size
    )
print("✓ EXACT evaluation complete")

# Print memory summary
memory_tracker.print_summary()

# Compare and save
print("\n" + "="*70)
print("ACCURACY COMPARISON")
print("="*70)

def get_metric_value(task_result):
    """Extract the primary accuracy metric from task results"""
    if 'acc_norm,none' in task_result:
        return task_result['acc_norm,none']
    elif 'acc,none' in task_result:
        return task_result['acc,none']
    elif 'acc_norm' in task_result:
        return task_result['acc_norm']
    elif 'acc' in task_result:
        return task_result['acc']
    else:
        return next((v for k, v in task_result.items() if isinstance(v, (int, float)) and k != 'alias'), 0)

results_data = []
print(f"\n{'Task':<20} {'Baseline':<12} {'EXACT':<12} {'Drop':<12}")
print("-" * 70)

for task in args.tasks.split(','):
    if baseline_results is not None:
        baseline_acc = get_metric_value(baseline_results['results'][task])
        exact_acc = get_metric_value(exact_results['results'][task])
        drop = (baseline_acc - exact_acc) * 100
        
        print(f"{task:<20} {baseline_acc*100:>6.2f}%      {exact_acc*100:>6.2f}%      {drop:>+6.2f}%")
        
        results_data.append({
            'task': task,
            'baseline_acc': baseline_acc * 100,
            'exact_acc': exact_acc * 100,
            'accuracy_drop': drop
        })
    else:
        exact_acc = get_metric_value(exact_results['results'][task])
        print(f"{task:<20} {'SKIPPED':<12} {exact_acc*100:>6.2f}%      {'N/A':<12}")
        
        results_data.append({
            'task': task,
            'baseline_acc': 'N/A',
            'exact_acc': exact_acc * 100,
            'accuracy_drop': 'N/A'
        })

# Save CSV
if results_data:
    model_name = args.model.replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = 'artifacts/results/model_accuracy_results'
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, f"exact_results_{model_name}_{timestamp}.csv")

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['task', 'baseline_acc', 'exact_acc', 'accuracy_drop']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\nAccuracy results saved to: {csv_filename}")
    
    # Save memory metrics
    memory_csv = os.path.join(results_dir, f"memory_usage_{model_name}_{timestamp}.csv")
    memory_summary = memory_tracker.get_summary_dict()
    memory_summary.update(model_comparison)
    memory_summary['model'] = args.model
    memory_summary['tasks'] = args.tasks
    memory_summary['batch_size'] = args.batch_size
    memory_summary['skip_baseline'] = args.skip_baseline
    
    with open(memory_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=memory_summary.keys())
        writer.writeheader()
        writer.writerow(memory_summary)
    
    print(f"Memory metrics saved to: {memory_csv}")