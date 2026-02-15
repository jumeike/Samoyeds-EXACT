import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import argparse
import csv
from datetime import datetime

from mixtral.modeling_mixtral import MixtralForCausalLM
from mixtral.configuration_mixtral import MixtralConfig
from exact_weight_loader_mixtral import load_pretrained_weights_into_samoyeds
from exact_core import apply_exact_to_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mistralai/Mixtral-8x7B-v0.1')
parser.add_argument('--profiling_json', default='artifacts/exact_profiling/mixtral-8x7b-v0.1_real_e8_layer.json')
parser.add_argument('--tasks', default='mmlu,hellaswag,arc_challenge,winogrande,openbookqa,gsm8k')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline evaluation')
args = parser.parse_args()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

# Evaluate official pretrained model as baseline
# 1. Load baseline to GPU, evaluate, immediately free
if not args.skip_baseline:
    print("="*70)
    print("EVALUATING OFFICIAL PRETRAINED MODEL (BASELINE)")
    print("="*70)
    print(f"Loading {args.model}...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    baseline_model.eval()
    
    print("Evaluating baseline...")
    baseline_lm = HFLM(pretrained=baseline_model, tokenizer=tokenizer)
    baseline_results = evaluator.simple_evaluate(
        model=baseline_lm,
        tasks=args.tasks.split(','),
        batch_size=args.batch_size
    )
    print("✓ Baseline evaluation complete")
    
    # Free GPU memory
    del baseline_model, baseline_lm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("✓ Baseline complete, memory freed")
else:
    baseline_results = None

# Create Samoyeds Mixtral model and load pretrained weights
# 2. Load to CPU, compress on CPU, move compressed to GPU
print("\n" + "="*70)
print("EVALUATING SAMOYEDS MIXTRAL WITH EXACT COMPRESSION")
print("="*70)
config = MixtralConfig.from_pretrained(args.model, trust_remote_code=True)

# Print model configuration
print(f"Mixtral-8x7B Configuration:")
print(f"  Model size: {config.num_hidden_layers} layers, {config.hidden_size} hidden dim")
print(f"  MoE: {config.num_local_experts} experts, top-{config.num_experts_per_tok} routing")
print(f"  Attention: {config.num_attention_heads} Q heads, {config.num_key_value_heads} KV heads (GQA)")
print(f"  FFN: {config.intermediate_size} intermediate dim")
print(f"  Context: {config.max_position_embeddings} tokens (sliding window: {config.sliding_window})")

# Create Samoyeds Mixtral model
print("Creating Samoyeds Mixtral model...")
# Note: You may need to create a full MixtralForCausalLM wrapper with Samoyeds MoE
# For now using the standard Mixtral model structure
samoyeds_model = MixtralForCausalLM(config).half()

# Load pretrained weights
print(f"Loading pretrained weights from {args.model}...")
samoyeds_model = load_pretrained_weights_into_samoyeds(samoyeds_model, args.model)
# samoyeds_model = samoyeds_model.cuda()
# samoyeds_model.eval()
print("✓ Weights loaded")

# Convert MoE blocks to Samoyeds sparse format
print("\nConverting MoE blocks to Samoyeds format...")
from mixtral.modeling_mixtral import MixtralSparseMoeBlock
from mixtral.modeling_mixtral_samoyeds import SSMixtralSparseMoeBlock

for layer in samoyeds_model.model.layers:
    if isinstance(layer.block_sparse_moe, MixtralSparseMoeBlock):
        original_moe = layer.block_sparse_moe
        # Convert to Samoyeds format - pass the original MoE block
        ss_moe = SSMixtralSparseMoeBlock(original_moe)
        # Weights are already copied in __init__, just replace
        layer.block_sparse_moe = ss_moe
print("✓ Converted to Samoyeds format")

# Apply EXACT compression on CPU
print(f"\nApplying EXACT compression from {args.profiling_json}...")
exact_model = apply_exact_to_model(samoyeds_model, args.profiling_json, dense_model=samoyeds_model)
# exact_model = exact_model.cuda()
# exact_model.eval()
print("✓ EXACT compression applied")

# Free uncompressed model
del samoyeds_model
import gc
gc.collect()

# Evaluate EXACT model
# Move compressed model to GPU
print("Moving compressed model to GPU...")
exact_model = exact_model.cuda()

# CRITICAL: Ensure all sparse tensors are on GPU
for layer in exact_model.model.layers:
    if hasattr(layer, 'block_sparse_moe'):
        for expert in layer.block_sparse_moe.experts:
            for module_name in ['w1', 'w2', 'w3']:
                if hasattr(expert, module_name):
                    module = getattr(expert, module_name)
                    if hasattr(module, 'indices') and hasattr(module, 'metadata'):
                        module.indices = module.indices.cuda()
                        module.metadata = module.metadata.cuda()

exact_model.eval()
print("✓ Model loaded to GPU")

print("\nEvaluating EXACT compressed model...")
exact_lm = HFLM(pretrained=exact_model, tokenizer=tokenizer)
exact_results = evaluator.simple_evaluate(
    model=exact_lm,
    tasks=args.tasks.split(','),
    batch_size=args.batch_size
)
print("✓ EXACT evaluation complete")

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
    results_dir = 'artifacts/results'
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, f"exact_results_{model_name}_{timestamp}.csv")

    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['task', 'baseline_acc', 'exact_acc', 'accuracy_drop']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\nResults saved to: {csv_filename}")
