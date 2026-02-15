#!/usr/bin/env python3
import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import argparse
import csv
from datetime import datetime
import gc
torch.cuda.empty_cache()
gc.collect()

# from qwen2_moe.modeling_qwen2_moe_samoyeds import SSQwen2MoeForCausalLM
from qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from exact_weight_loader_qwen import load_pretrained_weights_into_samoyeds
from exact_core import apply_exact_to_model, apply_uniform_samoyeds_sparsification

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Qwen/Qwen1.5-MoE-A2.7B')
parser.add_argument('--method', choices=['baseline', 'uniform_samoyeds', 'exact', 'test_pruning_only', 'test_hf_direct', 'test_custom_dense', 'test_random_weights', 'test_architecture_diff'], required=True,
                   help='baseline=dense, uniform_samoyeds=uniform 2:4, exact=adaptive compression')
parser.add_argument('--profiling_json', default='artifacts/exact_profiling/qwen1.5-moe-a2.7b_real_e60_layer.json',
                   help='Required for EXACT method')
parser.add_argument('--use_modelopt', action='store_true', help='Use ModelOpt ASP for pruning')
parser.add_argument('--tasks', default='mmlu,hellaswag,arc_challenge,winogrande')
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

print("="*70)
print(f"EVALUATION METHOD: {args.method.upper()}")
print("="*70)

if args.method == 'baseline':
    # Dense baseline - no compression
    print(f"Loading dense model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    model.eval()
    print("✓ Dense model loaded")

elif args.method == 'uniform_samoyeds':
    # Uniform 2:4 baseline - all experts pruned equally
    print("Creating dense model for weight extraction...")
    config = Qwen2MoeConfig.from_pretrained(args.model, trust_remote_code=True)
    dense_model = SSQwen2MoeForCausalLM(config, use_dense=True).half()
    dense_model = load_pretrained_weights_into_samoyeds(dense_model, args.model)
    print("✓ Dense weights loaded")
    
    print("Creating Samoyeds model...")
    samoyeds_model = SSQwen2MoeForCausalLM(config, use_dense=False).half()
    
    print(f"Applying uniform 2:4 sparsity (ModelOpt={args.use_modelopt})...")
    model = apply_uniform_samoyeds_sparsification(
        samoyeds_model, 
        dense_model, 
        use_modelopt=args.use_modelopt
    )
    
    # Free dense model
    del dense_model, samoyeds_model
    gc.collect()
    
    print("Moving to GPU...")
    model = model.cuda()
    model.eval()
    print("✓ Uniform Samoyeds model ready")

elif args.method == 'exact':
    # EXACT adaptive compression
    if args.profiling_json is None:
        raise ValueError("--profiling_json required for EXACT method")
    
    print("Creating dense model for weight extraction...")
    config = Qwen2MoeConfig.from_pretrained(args.model, trust_remote_code=True)
    dense_model = SSQwen2MoeForCausalLM(config, use_dense=True).half()
    dense_model = load_pretrained_weights_into_samoyeds(dense_model, args.model)
    print("✓ Dense weights loaded")
    
    print("Creating Samoyeds model...")
    samoyeds_model = SSQwen2MoeForCausalLM(config, use_dense=False).half()
    
    print(f"Applying EXACT adaptive compression from {args.profiling_json}...")
    print(f"  ModelOpt: {args.use_modelopt}")
    model = apply_exact_to_model(
        samoyeds_model, 
        args.profiling_json, 
        dense_model=dense_model,
        use_modelopt=args.use_modelopt
    )
    
    # Free uncompressed models
    del dense_model, samoyeds_model
    gc.collect()
    
    print("Moving to GPU...")
    model = model.cuda()
    model.eval()
    print("✓ EXACT model ready")

elif args.method == 'test_pruning_only':
    # Test pruning WITHOUT Samoyeds - just magnitude prune in-place
    print("Testing pruning accuracy without Samoyeds conversion...")
    config = Qwen2MoeConfig.from_pretrained(args.model, trust_remote_code=True)
    model = SSQwen2MoeForCausalLM(config, use_dense=True).half()  # Keep dense
    model = load_pretrained_weights_into_samoyeds(model, args.model)
    
    # Prune weights in-place (no Samoyeds format)
    from exact_core.samoyed_converter import _prune_24_magnitude
    for layer in model.model.layers:
        if hasattr(layer.mlp, 'experts'):
            for expert in layer.mlp.experts:
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    weight = getattr(expert, name).weight
                    pruned = _prune_24_magnitude(weight.data)
                    getattr(expert, name).weight.data = pruned.half()
    
    model = model.cuda()
    model.eval()
    print("✓ Pruned model ready (dense execution)")

elif args.method == 'test_hf_direct':
    # Load HF model, prune in-place, NO custom weight loader
    print("Testing with direct HF loading...")
    from transformers import Qwen2MoeForCausalLM  # Official HF class
    
    model = Qwen2MoeForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    
    # Prune in-place
    from exact_core.samoyed_converter import _prune_24_magnitude, _prune_14_magnitude, _prune_with_modelopt_24
    for layer in model.model.layers:
        if hasattr(layer.mlp, 'experts'):
            for expert in layer.mlp.experts:
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    weight = getattr(expert, name).weight
                    # Prune on CPU to avoid OOM, then copy back in-place
                    # pruned_cpu = _prune_24_magnitude(weight.data, return_to_device=False)
                    pruned_cpu = _prune_with_modelopt_24(weight.data)
                    getattr(expert, name).weight.data.copy_(pruned_cpu.to(weight.device, dtype=weight.dtype))
    
    model.eval()

elif args.method == 'test_custom_dense':
    # Test custom model + weight loader WITHOUT pruning
    print("Testing custom dense model with weight loader (no pruning)...")
    # config = Qwen2MoeConfig.from_pretrained(args.model, trust_remote_code=True)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = SSQwen2MoeForCausalLM(config, use_dense=True).half()
    model = load_pretrained_weights_into_samoyeds(model, args.model)
    print("✓ Custom dense model loaded")
    
    model = model.cuda()
    model.eval()
    print("✓ Custom dense model ready (no pruning)")

elif args.method == 'test_random_weights':
    # Test custom model with RANDOM weights (no loading)
    print("Testing SSQwen2MoeForCausalLM architecture with random weights...")
    # config = Qwen2MoeConfig.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = SSQwen2MoeForCausalLM(config, use_dense=True).half()
    # NO weight loading - keep random initialization
    print("✓ Random model created (no weight loading)")
    
    model = model.cuda()
    model.eval()
    print("✓ Random model ready")

elif args.method == 'test_architecture_diff':
    print("Comparing HF vs Custom architecture outputs...")
    
    # Test input
    test_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    
    # Load HF official model (CPU, float32)
    print("Loading HF model...")
    from transformers import Qwen2MoeForCausalLM
    hf_model = Qwen2MoeForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map='cpu',
        trust_remote_code=True
    )
    hf_model.eval()
    
    # Load custom model with same weights (CPU, float32) - BEFORE deleting HF model
    print("Loading custom model...")
    config = Qwen2MoeConfig.from_pretrained(args.model, trust_remote_code=True)
    custom_model = SSQwen2MoeForCausalLM(config, use_dense=True).float()
    custom_model = load_pretrained_weights_into_samoyeds(custom_model, args.model)
    custom_model.eval()
    
    # ADD DEBUG HOOKS HERE (after both models loaded)
    hf_moe_outs = []
    custom_moe_outs = []
    
    original_hf_forward = hf_model.model.layers[0].mlp.forward
    original_custom_forward = custom_model.model.layers[0].mlp.forward
    
    def hf_wrapper(x):
        out = original_hf_forward(x)
        hf_moe_outs.append(out[0].detach().clone() if isinstance(out, tuple) else out.detach().clone())
        return out
    
    def custom_wrapper(x):
        out = original_custom_forward(x)
        custom_moe_outs.append(out[0].detach().clone() if isinstance(out, tuple) else out.detach().clone())
        return out
    
    hf_model.model.layers[0].mlp.forward = hf_wrapper
    custom_model.model.layers[0].mlp.forward = custom_wrapper

    # ADD HERE:
    print("\nWeight verification (Layer 0 MoE):")
    hf_moe = hf_model.model.layers[0].mlp
    custom_moe = custom_model.model.layers[0].mlp
    print(f"Gate: {(hf_moe.gate.weight - custom_moe.gate.weight).abs().max():.8f}")
    print(f"Expert 0 gate_proj: {(hf_moe.experts[0].gate_proj.weight - custom_moe.experts[0].gate_proj.weight).abs().max():.8f}")
    print(f"Shared expert: {(hf_moe.shared_expert.gate_proj.weight - custom_moe.shared_expert.gate_proj.weight).abs().max():.8f}")

    print(f"\nHF MoE type: {type(hf_moe)}")
    print(f"Custom MoE type: {type(custom_moe)}")
    print(f"HF config norm_topk_prob: {hf_model.config.norm_topk_prob}")
    print(f"Custom config norm_topk_prob: {custom_model.config.norm_topk_prob}")
    
    # Get HF output
    print("Running HF model...")
    with torch.no_grad():
        hf_out = hf_model(test_input).logits
    
    print(f"HF output shape: {hf_out.shape}, mean: {hf_out.mean():.6f}")
    
    # Get custom output
    print("Running custom model...")
    with torch.no_grad():
        custom_out = custom_model(test_input).logits
    
    print(f"Custom output shape: {custom_out.shape}, mean: {custom_out.mean():.6f}")
    
    # Compare
    print(f"\n{'='*70}")
    print("ARCHITECTURE COMPARISON")
    print(f"{'='*70}")
    print(f"Output diff (abs mean): {(hf_out - custom_out).abs().mean():.6f}")
    print(f"Output diff (max): {(hf_out - custom_out).abs().max():.6f}")
    print(f"\nLayer 0 MoE output diff: {(hf_moe_outs[0] - custom_moe_outs[0]).abs().mean():.6f}")
    print(f"Layer 0 MoE output max diff: {(hf_moe_outs[0] - custom_moe_outs[0]).abs().max():.6f}")
    
    import sys
    sys.exit(0)

# elif args.method == 'test_custom_dense':
#     print("Testing custom dense model with weight loader (no pruning)...")
#     config = Qwen2MoeConfig.from_pretrained(args.model, trust_remote_code=True)

#     # Compare HF vs custom on CPU first to avoid GPU OOM
#     print("\n" + "="*70)
#     print("TESTING FORWARD PASS")
#     print("="*70)

#     from transformers import Qwen2MoeForCausalLM
#     hf_model = Qwen2MoeForCausalLM.from_pretrained(
#         args.model,
#         torch_dtype=torch.float32,
#         device_map='cpu',
#         trust_remote_code=True,
#         low_cpu_mem_usage=True
#     ).eval()

#     model_cpu = SSQwen2MoeForCausalLM(config, use_dense=True).float()
#     model_cpu = load_pretrained_weights_into_samoyeds(model_cpu, args.model)
#     model_cpu.eval()

#     test_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

#     with torch.no_grad():
#         hf_out = hf_model(test_input).logits
#         custom_out = model_cpu(test_input).logits

#     print(f"HF output mean: {hf_out.mean():.6f}")
#     print(f"Custom output mean: {custom_out.mean():.6f}")
#     print(f"Absolute difference: {(hf_out - custom_out).abs().mean():.6f}")
#     print(f"Max difference: {(hf_out - custom_out).abs().max():.6f}")

#     if (hf_out - custom_out).abs().mean() > 0.1:
#         print("\n❌ FORWARD PASS MISMATCH DETECTED")
#     else:
#         print("\n✓ Forward pass matches HF")
#     print("="*70 + "\n")

#     # Free CPU models before constructing the GPU model
#     del hf_model, model_cpu
#     gc.collect()

#     print("Rebuilding custom dense model on GPU...")
#     model = SSQwen2MoeForCausalLM(config, use_dense=True).half()
#     model = load_pretrained_weights_into_samoyeds(model, args.model)
#     model = model.cuda()
#     model.eval()
#     print("✓ Custom dense model ready (GPU)")

# Evaluate
print(f"\nEvaluating on tasks: {args.tasks}")
lm = HFLM(pretrained=model, tokenizer=tokenizer)
results = evaluator.simple_evaluate(
    model=lm,
    tasks=args.tasks.split(','),
    batch_size=args.batch_size
)
print("✓ Evaluation complete")

# Print results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

def get_metric_value(task_result):
    """Extract primary accuracy metric"""
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
print(f"\n{'Task':<20} {'Accuracy':<12}")
print("-" * 35)

for task in args.tasks.split(','):
    acc = get_metric_value(results['results'][task])
    print(f"{task:<20} {acc*100:>6.2f}%")
    
    results_data.append({
        'method': args.method,
        'use_modelopt': args.use_modelopt,
        'task': task,
        'accuracy': acc * 100
    })

# Save results
# model_name = args.model.replace('/', '_')
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# results_dir = 'artifacts/results/compression_comparison'
# os.makedirs(results_dir, exist_ok=True)

# csv_filename = os.path.join(results_dir, f"{args.method}_{model_name}_{timestamp}.csv")
# with open(csv_filename, 'w', newline='') as csvfile:
#     fieldnames = ['method', 'use_modelopt', 'task', 'accuracy']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(results_data)

# print(f"\nResults saved to: {csv_filename}")