#!/usr/bin/env python3
"""
EXACT Evaluation: Monkey-patch HF's Qwen2MoeSparseMoeBlock.forward
to use Samoyeds sparse kernels directly (no wrappers)
"""
import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import argparse
import json

from module.linear.SSTransLinear import SSTransLinear
from module.linear.SSFusedSiluTransLinear import SSFusedSiluTransLinear
from module.linear.SPDenseWeightedTransLinear import SPDenseWeightedLinear
from module.linear.SPDenseTransLinear import SPDenseTransLinear
from exact_core.samoyed_converter import convert_to_samoyed_format, _prune_24_magnitude, _prune_14_magnitude

DEBUG_PARITY = False
DEBUG_EXPERT_IDX = 0
DEBUG_MAX_TOKENS = 16
DEBUG_NO_ROUTING = False
DEBUG_PARITY_STAGE = "all"  # all|gate|up|combined|down


def create_samoyeds_moe_forward(original_forward):
    """
    Create new forward function that handles Samoyeds sparse experts.
    Replaces HF's Qwen2MoeSparseMoeBlock.forward
    """
    def samoyeds_moe_forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Router (unchanged)
        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        # Process experts
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                continue
            
            current_state = hidden_states_flat  # Samoyeds kernels expect full hidden_states
            
            # Check if expert is Samoyeds sparse
            if hasattr(expert, '_is_samoyeds'):
                # Samoyeds expert: outputs are [out_features, padded_batch]
                input_idx = top_x
                
                # gate * up (both return [out_features, padded_batch])
                gate_out = expert.gate_proj(current_state, input_idx)
                up_out = expert.up_proj(current_state, input_idx)
                combined = gate_out * up_out
                
                # down_proj: use routing weights transposed to match kernel expectation
                route_weights_subset = routing_weights[top_x, idx].unsqueeze(-1)
                down_out = expert.down_proj(combined, route_weights_subset.T)
                current_hidden_states = down_out[0:input_idx.shape[0], :]

                # Optional parity check against dense math
                if DEBUG_PARITY and not hasattr(self, '_debug_done'):
                    if getattr(expert, '_expert_idx', None) == DEBUG_EXPERT_IDX and top_x.numel() > 0:
                        sel = torch.arange(top_x.numel(), device=top_x.device)[:DEBUG_MAX_TOKENS]
                        tok = top_x[sel]
                        idx_sel = idx[sel]
                        gate_w = expert._dense_gate_weight.to(hidden_states_flat.device, dtype=hidden_states_flat.dtype)
                        up_w = expert._dense_up_weight.to(hidden_states_flat.device, dtype=hidden_states_flat.dtype)
                        down_w = expert._dense_down_weight.to(hidden_states_flat.device, dtype=hidden_states_flat.dtype)
                        dense_gate = F.linear(hidden_states_flat[tok], gate_w)
                        dense_up = F.linear(hidden_states_flat[tok], up_w)
                        dense_combined = expert.act_fn(dense_gate) * dense_up
                        dense_out = F.linear(dense_combined, down_w)
                        if not DEBUG_NO_ROUTING:
                            dense_out = dense_out * routing_weights[tok, idx_sel].unsqueeze(-1)
                            kernel_out = current_hidden_states[:tok.numel(), :]
                        else:
                            kernel_out = down_out[0:tok.numel(), :]

                        if DEBUG_PARITY_STAGE in ("all", "gate"):
                            diff = (gate_out[:, :tok.numel()].T - dense_gate).abs()
                            print(f"[DEBUG_PARITY][gate] expert={DEBUG_EXPERT_IDX} tok={tok.numel()} "
                                  f"max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
                        if DEBUG_PARITY_STAGE in ("all", "up"):
                            diff = (up_out[:, :tok.numel()].T - dense_up).abs()
                            print(f"[DEBUG_PARITY][up] expert={DEBUG_EXPERT_IDX} tok={tok.numel()} "
                                  f"max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
                        if DEBUG_PARITY_STAGE in ("all", "combined"):
                            diff = (combined[:, :tok.numel()].T - dense_combined).abs()
                            print(f"[DEBUG_PARITY][combined] expert={DEBUG_EXPERT_IDX} tok={tok.numel()} "
                                  f"max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
                        if DEBUG_PARITY_STAGE in ("all", "down"):
                            diff = (kernel_out - dense_out).abs()
                            print(f"[DEBUG_PARITY][down] expert={DEBUG_EXPERT_IDX} tok={tok.numel()} "
                                  f"max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
                        self._debug_done = True
            else:
                # Dense HF expert
                current_hidden_states = expert(current_state) * routing_weights[top_x, idx, None]
            
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        # Shared expert
        if hasattr(self.shared_expert, '_is_samoyeds'):
            # Samoyeds shared expert uses SPDenseTransLinear (no routing weights)
            shared_gate = self.shared_expert.gate_proj(hidden_states_flat)
            shared_up = self.shared_expert.up_proj(hidden_states_flat)
            shared_combined = self.shared_expert.act_fn(shared_gate) * shared_up
            shared_expert_output = self.shared_expert.down_proj(shared_combined)
        else:
            shared_expert_output = self.shared_expert(hidden_states_flat)
        
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
    
    return samoyeds_moe_forward


def replace_expert_samoyeds_uniform(expert):
    """Replace expert with uniform 2:4 Samoyeds"""
    # Prune uniformly to 2:4
    expert.gate_proj.weight.data = _prune_24_magnitude(expert.gate_proj.weight.data)
    expert.up_proj.weight.data = _prune_24_magnitude(expert.up_proj.weight.data)
    expert.down_proj.weight.data = _prune_24_magnitude(expert.down_proj.weight.data)

    # Save dense weights for parity checks
    if DEBUG_PARITY and getattr(expert, '_expert_idx', None) == DEBUG_EXPERT_IDX:
        # Keep debug weights on CPU to avoid GPU OOM
        expert._dense_gate_weight = expert.gate_proj.weight.detach().to("cpu").clone()
        expert._dense_up_weight = expert.up_proj.weight.detach().to("cpu").clone()
        expert._dense_down_weight = expert.down_proj.weight.detach().to("cpu").clone()
    
    # Create Samoyeds layers with explicit 2:4 packing
    sparsity_type = 'hot'
    n, m = 2, 4

    gate_sparse = SSFusedSiluTransLinear(
        original=None,
        in_features=expert.gate_proj.in_features,
        out_features=expert.gate_proj.out_features,
        skip_sparsifier=True
    )
    gate_sparse.N, gate_sparse.M = n, m
    gate_val, gate_idx, gate_meta = convert_to_samoyed_format(expert.gate_proj.weight.data, sparsity_type)
    gate_sparse.weight = nn.Parameter(gate_val)
    gate_sparse.indices = nn.Parameter(gate_idx, requires_grad=False)
    gate_sparse.metadata = nn.Parameter(gate_meta, requires_grad=False)
    gate_sparse.original_out_features = expert.gate_proj.out_features
    gate_sparse.original_in_features = expert.gate_proj.in_features

    up_sparse = SSTransLinear(
        original=None,
        in_features=expert.up_proj.in_features,
        out_features=expert.up_proj.out_features,
        skip_sparsifier=True
    )
    up_sparse.N, up_sparse.M = n, m
    up_val, up_idx, up_meta = convert_to_samoyed_format(expert.up_proj.weight.data, sparsity_type)
    up_sparse.weight = nn.Parameter(up_val)
    up_sparse.indices = nn.Parameter(up_idx, requires_grad=False)
    up_sparse.metadata = nn.Parameter(up_meta, requires_grad=False)
    up_sparse.original_out_features = expert.up_proj.out_features
    up_sparse.original_in_features = expert.up_proj.in_features

    down_sparse = SPDenseWeightedLinear(
        original=None,
        in_features=expert.down_proj.in_features,
        out_features=expert.down_proj.out_features,
        skip_sparsifier=True
    )
    down_sparse.N, down_sparse.M = n, m
    down_val, down_idx, down_meta = convert_to_samoyed_format(expert.down_proj.weight.data, sparsity_type)
    down_sparse.weight = nn.Parameter(down_val)
    down_sparse.indices = nn.Parameter(down_idx, requires_grad=False)
    down_sparse.metadata = nn.Parameter(down_meta, requires_grad=False)
    down_sparse.original_out_features = expert.down_proj.out_features
    down_sparse.original_in_features = expert.down_proj.in_features

    expert.gate_proj = gate_sparse
    expert.up_proj = up_sparse
    expert.down_proj = down_sparse
    expert._is_samoyeds = True


def apply_samoyeds_uniform(model):
    """Apply uniform 2:4 Samoyeds compression (no pruning, uses built-in sparsifier)"""
    print("Uniform 2:4 Samoyeds compression...")
    
    for l_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, 'experts'):
            continue
        
        for e_idx, expert in enumerate(layer.mlp.experts):
            expert._expert_idx = e_idx
            replace_expert_samoyeds_uniform(expert)
        
        # Monkey-patch MoE forward
        layer.mlp.forward = create_samoyeds_moe_forward(layer.mlp.forward).__get__(layer.mlp)
        
        if l_idx % 5 == 0:
            print(f"  L{l_idx}")
    
    print("✓ Uniform Samoyeds applied")
    return model


def replace_expert_exact(expert, n, m):
    """Replace expert's Linear layers with Samoyeds sparse, mark as Samoyeds"""
    sparsity_type = 'hot' if n == 2 else 'cold'
    
    # Prune
    if n == 2:
        expert.gate_proj.weight.data = _prune_24_magnitude(expert.gate_proj.weight.data)
        expert.up_proj.weight.data = _prune_24_magnitude(expert.up_proj.weight.data)
        expert.down_proj.weight.data = _prune_24_magnitude(expert.down_proj.weight.data)
    else:
        expert.gate_proj.weight.data = _prune_14_magnitude(expert.gate_proj.weight.data)
        expert.up_proj.weight.data = _prune_14_magnitude(expert.up_proj.weight.data)
        expert.down_proj.weight.data = _prune_14_magnitude(expert.down_proj.weight.data)

    # Save dense weights for parity checks
    if DEBUG_PARITY and getattr(expert, '_expert_idx', None) == DEBUG_EXPERT_IDX:
        # Keep debug weights on CPU to avoid GPU OOM
        expert._dense_gate_weight = expert.gate_proj.weight.detach().to("cpu").clone()
        expert._dense_up_weight = expert.up_proj.weight.detach().to("cpu").clone()
        expert._dense_down_weight = expert.down_proj.weight.detach().to("cpu").clone()
    
    # Create Samoyeds layers - pass None as original, provide dimensions
    gate_sparse = SSFusedSiluTransLinear(
        original=None,
        in_features=expert.gate_proj.in_features,
        out_features=expert.gate_proj.out_features,
        skip_sparsifier=True
    )
    gate_sparse.N, gate_sparse.M = n, m
    gate_val, gate_idx, gate_meta = convert_to_samoyed_format(expert.gate_proj.weight.data, sparsity_type)
    gate_sparse.weight = nn.Parameter(gate_val)
    gate_sparse.indices = nn.Parameter(gate_idx, requires_grad=False)
    gate_sparse.metadata = nn.Parameter(gate_meta, requires_grad=False)
    gate_sparse.original_out_features = expert.gate_proj.out_features
    gate_sparse.original_in_features = expert.gate_proj.in_features
    
    up_sparse = SSTransLinear(
        original=None,
        in_features=expert.up_proj.in_features,
        out_features=expert.up_proj.out_features,
        skip_sparsifier=True
    )
    up_sparse.N, up_sparse.M = n, m
    up_val, up_idx, up_meta = convert_to_samoyed_format(expert.up_proj.weight.data, sparsity_type)
    up_sparse.weight = nn.Parameter(up_val)
    up_sparse.indices = nn.Parameter(up_idx, requires_grad=False)
    up_sparse.metadata = nn.Parameter(up_meta, requires_grad=False)
    up_sparse.original_out_features = expert.up_proj.out_features
    up_sparse.original_in_features = expert.up_proj.in_features
    
    down_sparse = SPDenseWeightedLinear(
        original=None,
        in_features=expert.down_proj.in_features,
        out_features=expert.down_proj.out_features,
        skip_sparsifier=True
    )
    down_sparse.N, down_sparse.M = n, m
    down_val, down_idx, down_meta = convert_to_samoyed_format(expert.down_proj.weight.data, sparsity_type)
    down_sparse.weight = nn.Parameter(down_val)
    down_sparse.indices = nn.Parameter(down_idx, requires_grad=False)
    down_sparse.metadata = nn.Parameter(down_meta, requires_grad=False)
    down_sparse.original_out_features = expert.down_proj.out_features
    down_sparse.original_in_features = expert.down_proj.in_features
    
    expert.gate_proj = gate_sparse
    expert.up_proj = up_sparse
    expert.down_proj = down_sparse
    expert._is_samoyeds = True  # Marker


def apply_exact_samoyeds(model, profiling_json):
    """Apply EXACT and monkey-patch MoE forward"""
    with open(profiling_json) as f:
        prof = json.load(f)
    
    if 'layers' not in prof:
        raise ValueError(f"Expected 'layers' key. Got: {list(prof.keys())}")
    
    layers_data = prof['layers']
    print(f"EXACT compression for {len(layers_data)} layers...")
    
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
            expert._expert_idx = e_idx
            n, m = (2, 4) if e_idx in hot_experts else (1, 4)
            replace_expert_exact(expert, n, m)
        
        # Monkey-patch MoE forward
        layer.mlp.forward = create_samoyeds_moe_forward(layer.mlp.forward).__get__(layer.mlp)
        
        if l_idx % 5 == 0:
            print(f"  L{l_idx}: {len(hot_experts)} hot, {len(cold_experts)} cold")
    
    print("✓ EXACT applied with monkey-patched forward")
    return model


# CLI
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Qwen/Qwen1.5-MoE-A2.7B')
parser.add_argument('--method', choices=['baseline', 'uniform_samoyeds', 'exact', 'test_hf_direct'], required=True)
parser.add_argument('--profiling_json', default='artifacts/exact_profiling/qwen1.5-moe-a2.7b_real_e60_layer.json')
parser.add_argument('--tasks', default='mmlu_astronomy')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--debug_parity', action='store_true')
parser.add_argument('--debug_expert', type=int, default=0)
parser.add_argument('--debug_max_tokens', type=int, default=16)
parser.add_argument('--debug_no_routing', action='store_true')
parser.add_argument('--debug_parity_stage', default='all', choices=['all', 'gate', 'up', 'combined', 'down'])
args = parser.parse_args()

DEBUG_PARITY = args.debug_parity
DEBUG_EXPERT_IDX = args.debug_expert
DEBUG_MAX_TOKENS = args.debug_max_tokens
DEBUG_NO_ROUTING = args.debug_no_routing
DEBUG_PARITY_STAGE = args.debug_parity_stage

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

print("="*70)
print(f"METHOD: {args.method.upper()}")
print("="*70)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)
print("✓ Loaded")

if args.method == 'uniform_samoyeds':
    model = apply_samoyeds_uniform(model)
elif args.method == 'exact':
    model = apply_exact_samoyeds(model, args.profiling_json)
elif args.method == 'test_hf_direct':
    # Prune HF model in-place (no Samoyeds conversion)
    print("Testing with direct HF loading + pruning only...")
    for layer in model.model.layers:
        if hasattr(layer.mlp, 'experts'):
            for expert in layer.mlp.experts:
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    weight = getattr(expert, name).weight
                    pruned = _prune_24_magnitude(weight.data)
                    getattr(expert, name).weight.data.copy_(pruned.to(weight.device, dtype=weight.dtype))

model.eval()

print(f"\nEvaluating: {args.tasks}")
lm = HFLM(pretrained=model, tokenizer=tokenizer)
results = evaluator.simple_evaluate(model=lm, tasks=args.tasks.split(','), batch_size=args.batch_size)

print("\n" + "="*70)
print("RESULTS")
print("="*70)

def get_acc(r):
    for k in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc']:
        if k in r:
            return r[k]
    return 0

print(f"\n{'Task':<20} {'Accuracy'}")
print("-" * 35)
for task in args.tasks.split(','):
    print(f"{task:<20} {get_acc(results['results'][task])*100:>6.2f}%")
