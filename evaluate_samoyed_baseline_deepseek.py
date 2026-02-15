import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import argparse
import csv
from datetime import datetime

from deepseek.modeling_deepseek import DeepseekForCausalLM
from deepseek.configuration_deepseek import DeepseekConfig
from exact_weight_loader_deepseek import load_pretrained_weights_into_samoyeds
from memory_tracker import MemoryTracker, compare_model_sizes

parser = argparse.ArgumentParser(description='Evaluate Samoyed sparse MoE baseline against dense baseline')
parser.add_argument('--model', default='deepseek-ai/deepseek-moe-16b-base', help='HuggingFace model name')
parser.add_argument('--tasks', default='mmlu,hellaswag,arc_challenge,winogrande,openbookqa,gsm8k', help='Comma-separated list of tasks')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
parser.add_argument('--skip_dense', action='store_true', help='Skip dense baseline evaluation')
args = parser.parse_args()

# Initialize memory tracker
memory_tracker = MemoryTracker()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

dense_results = None

# Evaluate official pretrained dense model as baseline
if not args.skip_dense:
    print("="*70)
    print("EVALUATING DENSE BASELINE (HuggingFace Transformers)")
    print("="*70)
    print(f"Loading {args.model}...")
    dense_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    dense_model.eval()
    
    with memory_tracker.track('dense_inference'):
        print("Evaluating dense baseline...")
        dense_lm = HFLM(pretrained=dense_model, tokenizer=tokenizer)
        dense_results = evaluator.simple_evaluate(
            model=dense_lm,
            tasks=args.tasks.split(','),
            batch_size=args.batch_size
        )
    print("✓ Dense baseline evaluation complete")
    
    # Free GPU memory
    print("Freeing GPU memory...")
    dense_model = dense_model.cpu()
    del dense_model
    del dense_lm
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("✓ GPU memory cleared")

# Create and evaluate Samoyed baseline
print("\n" + "="*70)
print("EVALUATING SAMOYED BASELINE (Sparse MoE Implementation)")
print("="*70)

# Load configuration
config = DeepseekConfig.from_pretrained(args.model, trust_remote_code=True)

# Print model configuration
print(f"DeepSeek-MoE-16B Configuration:")
print(f"  Model size: {config.num_hidden_layers} layers, {config.hidden_size} hidden dim")
print(f"  MoE: {config.n_routed_experts} routed experts + {config.n_shared_experts} shared experts")
print(f"  Routing: top-{config.num_experts_per_tok}, every {config.moe_layer_freq} layer(s)")
print(f"  Attention: {config.num_attention_heads} heads (MHA)")
print(f"  FFN: dense={config.intermediate_size}, MoE={config.moe_intermediate_size} per expert")
print(f"  Context: {config.max_position_embeddings} tokens")

# Create DeepSeek models (dense for weights, sparse for evaluation after pruning)
print("\nCreating dense DeepSeek model for weight loading...")
dense_model = DeepseekForCausalLM(config).half()

print(f"Loading pretrained weights into dense model from {args.model}...")
dense_model = load_pretrained_weights_into_samoyeds(dense_model, args.model)
print("✓ Weights loaded into dense model")

print("\nCreating sparse DeepSeek model for baseline evaluation...")
samoyeds_model = DeepseekForCausalLM(config).half()

print(f"Loading pretrained weights into sparse model from {args.model}...")
samoyeds_model = load_pretrained_weights_into_samoyeds(samoyeds_model, args.model)
print("✓ Weights loaded into sparse model")

# Apply uniform 2:4 magnitude pruning to all experts (apples-to-apples with EXACT)
from exact_core.samoyed_baseline_utils import apply_uniform_sparsification

apply_uniform_sparsification(samoyeds_model, dense_model)
del dense_model
import gc
gc.collect()
# The dense DeepseekMoE already loaded from pretrained weights is what we evaluate

# Compare model sizes before evaluation
if not args.skip_dense:
    baseline_for_comparison = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True
    )
    model_comparison = compare_model_sizes(baseline_for_comparison, samoyeds_model)
    del baseline_for_comparison
    gc.collect()
else:
    model_comparison = {}

# Evaluate Samoyed baseline (with dense MoE, not sparse)
print("\nEvaluating Samoyed baseline (uniform 2:4 pruned MoE)...")
samoyeds_model = samoyeds_model.cuda()
samoyeds_model.eval()

with memory_tracker.track('samoyed_inference'):
    samoyeds_lm = HFLM(pretrained=samoyeds_model, tokenizer=tokenizer)
    samoyeds_results = evaluator.simple_evaluate(
        model=samoyeds_lm,
        tasks=args.tasks.split(','),
        batch_size=args.batch_size
    )
print("✓ Samoyed evaluation complete")

# Print memory summary
memory_tracker.print_summary()

# Compare and save results
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

if dense_results is not None:
    print(f"\n{'Task':<20} {'Dense':<12} {'Samoyed':<12} {'Drop':<12}")
    print("-" * 70)
    
    for task in args.tasks.split(','):
        dense_acc = get_metric_value(dense_results['results'][task])
        samoyed_acc = get_metric_value(samoyeds_results['results'][task])
        drop = (dense_acc - samoyed_acc) * 100
        
        print(f"{task:<20} {dense_acc*100:>6.2f}%      {samoyed_acc*100:>6.2f}%      {drop:>+6.2f}%")
        
        results_data.append({
            'task': task,
            'dense_acc': dense_acc * 100,
            'samoyed_acc': samoyed_acc * 100,
            'accuracy_drop': drop
        })
else:
    print(f"\n{'Task':<20} {'Samoyed':<12}")
    print("-" * 70)
    
    for task in args.tasks.split(','):
        samoyed_acc = get_metric_value(samoyeds_results['results'][task])
        print(f"{task:<20} {samoyed_acc*100:>6.2f}%")
        
        results_data.append({
            'task': task,
            'dense_acc': 'N/A',
            'samoyed_acc': samoyed_acc * 100,
            'accuracy_drop': 'N/A'
        })

# Save CSV
if results_data:
    model_name = args.model.replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = 'artifacts/results/model_accuracy_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save accuracy results
    csv_filename = os.path.join(results_dir, f"samoyed_baseline_{model_name}_{timestamp}.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['task', 'dense_acc', 'samoyed_acc', 'accuracy_drop']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)
    print(f"\nAccuracy results saved to: {csv_filename}")
    
    # Save memory metrics
    memory_csv = os.path.join(results_dir, f"memory_baseline_{model_name}_{timestamp}.csv")
    memory_summary = memory_tracker.get_summary_dict()
    memory_summary.update(model_comparison)
    memory_summary['model'] = args.model
    memory_summary['tasks'] = args.tasks
    memory_summary['batch_size'] = args.batch_size
    memory_summary['skip_dense'] = args.skip_dense
    
    with open(memory_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=memory_summary.keys())
        writer.writeheader()
        writer.writerow(memory_summary)
    print(f"Memory metrics saved to: {memory_csv}")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
print("\nSummary:")
print(f"  Dense baseline: {'Evaluated' if dense_results is not None else 'Skipped'}")
print(f"  Samoyed baseline: Evaluated")
print(f"  Tasks: {args.tasks}")
print(f"  Batch size: {args.batch_size}")
