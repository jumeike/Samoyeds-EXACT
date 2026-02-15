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
from exact_core import apply_exact_to_model
from memory_tracker import MemoryTracker, compare_model_sizes

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='deepseek-ai/deepseek-moe-16b-base')
parser.add_argument('--profiling_json', default='artifacts/exact_profiling/deepseek-moe-16b-base_real_e64_layer.json')
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

# Create Samoyeds DeepSeek model and load pretrained weights
print("\n" + "="*70)
print("EVALUATING SAMOYEDS DEEPSEEK WITH EXACT COMPRESSION")
print("="*70)
config = DeepseekConfig.from_pretrained(args.model, trust_remote_code=True)

# Print model configuration
print(f"DeepSeek-MoE-16B Configuration:")
print(f"  Model size: {config.num_hidden_layers} layers, {config.hidden_size} hidden dim")
print(f"  MoE: {config.n_routed_experts} routed experts + {config.n_shared_experts} shared experts")
print(f"  Routing: top-{config.num_experts_per_tok}, every {config.moe_layer_freq} layer(s)")
print(f"  Attention: {config.num_attention_heads} heads (MHA)")
print(f"  FFN: dense={config.intermediate_size}, MoE={config.moe_intermediate_size} per expert")
print(f"  Context: {config.max_position_embeddings} tokens")

# Create dense model for baseline evaluation (will load weights below)
print("Creating Dense DeepSeek model...")
dense_model = DeepseekForCausalLM(config).half()

# Create Samoyeds DeepSeek model
print("Creating Samoyeds DeepSeek model...")
samoyeds_model = DeepseekForCausalLM(config).half()

# Load pretrained weights into BOTH models
print(f"Loading pretrained weights from {args.model}...")
dense_model = load_pretrained_weights_into_samoyeds(dense_model, args.model)
samoyeds_model = load_pretrained_weights_into_samoyeds(samoyeds_model, args.model)
# print("✓ Weights loaded")

# Verify weight loading
# print("\n[DEBUG] Verifying weight loading...")
# hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True)
# hf_expert0 = hf_model.model.layers[1].mlp.experts[0].gate_proj.weight
# dense_expert0 = dense_model.model.layers[1].mlp.experts[0].gate_proj.weight
# sam_expert0 = samoyeds_model.model.layers[1].mlp.experts[0].gate_proj.weight
# print(f"HF vs Dense: {torch.allclose(hf_expert0, dense_expert0, rtol=1e-3)}")
# print(f"HF vs Samoyeds: {torch.allclose(hf_expert0, sam_expert0, rtol=1e-3)}")
# del hf_model

# Convert MoE blocks to Samoyeds sparse format
# IMPORTANT: Don't copy weights - SSDeepseekMLP already has spars structure
# EXACT will load dense weights from dense_model and sparsify them
print("\nConverting MoE blocks to Samoyeds format...")
from deepseek.modeling_deepseek import DeepseekMoE
from deepseek.modeling_deepseek_samoyeds import SSDeepseekMoE

# Replace MoE blocks
for layer in samoyeds_model.model.layers:
    if isinstance(layer.mlp, DeepseekMoE):
        # Convert to Samoyeds format (creates sparse structure)
        ss_moe = SSDeepseekMoE(layer.mlp.config, skip_sparsifier=True)
        # Copy gate weights (routing network - stays dense)
        ss_moe.gate.weight.data = layer.mlp.gate.weight.data.clone()
        # Copy shared experts if present (these stay dense too)
        if hasattr(layer.mlp, 'shared_experts') and hasattr(ss_moe, 'shared_experts'):
            ss_moe.shared_experts.gate_proj.weight.data = layer.mlp.shared_experts.gate_proj.weight.data.clone()
            ss_moe.shared_experts.up_proj.weight.data = layer.mlp.shared_experts.up_proj.weight.data.clone()
            ss_moe.shared_experts.down_proj.weight.data = layer.mlp.shared_experts.down_proj.weight.data.clone()
        # Copy routed experts
        for i, expert in enumerate(layer.mlp.experts):
            ss_moe.experts[i].gate_proj.weight.data = expert.gate_proj.weight.data.clone()
            ss_moe.experts[i].up_proj.weight.data = expert.up_proj.weight.data.clone()
            ss_moe.experts[i].down_proj.weight.data = expert.down_proj.weight.data.clone()
        # Replace the MoE block
        layer.mlp = ss_moe
# print("✓ Converted to Samoyeds format")

# Verify gate weight copying
# print("\n[DEBUG] Verifying gate/shared weights after conversion:")
# layer_idx = 1  # Check an MoE layer
# if isinstance(dense_model.model.layers[layer_idx].mlp, DeepseekMoE):
#     print(f"  Dense gate shape: {dense_model.model.layers[layer_idx].mlp.gate.weight.shape}")
#     print(f"  Sparse gate shape: {samoyeds_model.model.layers[layer_idx].mlp.gate.weight.shape}")
#     print(f"  Gates match: {torch.allclose(dense_model.model.layers[layer_idx].mlp.gate.weight, samoyeds_model.model.layers[layer_idx].mlp.gate.weight)}")
#     if hasattr(dense_model.model.layers[layer_idx].mlp, 'shared_experts'):
#         print(f"  Shared expert gate_proj match: {torch.allclose(dense_model.model.layers[layer_idx].mlp.shared_experts.gate_proj.weight, samoyeds_model.model.layers[layer_idx].mlp.shared_experts.gate_proj.weight)}")

# Apply EXACT compression (works on dense DeepseekMoE, will convert to sparse internally)
print(f"\nApplying EXACT compression from {args.profiling_json}...")
# CRITICAL: samoyeds_model still has dense DeepseekMoE - EXACT converts to sparse
exact_model = apply_exact_to_model(samoyeds_model, args.profiling_json, dense_model=dense_model)
print("✓ EXACT compression applied")

# DEBUG: Check if EXACT actually replaced weights
# print("\n[DEBUG] After EXACT compression:")
# first_expert = exact_model.model.layers[1].mlp.experts[0]
# print(f"  Expert type: {type(first_expert).__name__}")
# print(f"  gate_proj type: {type(first_expert.gate_proj).__name__}")
# if hasattr(first_expert.gate_proj, 'weight'):
#     print(f"  Weight shape: {first_expert.gate_proj.weight.shape}")
#     print(f"  Weight sum: {first_expert.gate_proj.weight.sum().item():.4f}")
#     print(f"  Weight mean: {first_expert.gate_proj.weight.mean().item():.6f}")
# if hasattr(first_expert.gate_proj, 'indices'):
#     print(f"  Has sparse format: indices shape {first_expert.gate_proj.indices.shape}")
#     print(f"  N:M = {first_expert.gate_proj.N}:{first_expert.gate_proj.M}")

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
    
    # Save accuracy results
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