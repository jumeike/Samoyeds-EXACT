#!/usr/bin/env python3
"""
Real routing profiler using actual model inference on calibration data
"""
import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm


def profile_real_routing(model_name, output_file, num_samples=500, max_seq_len=512):
    """Profile expert usage with real routing decisions"""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    
    # Enable router logits output if supported
    if hasattr(model.config, 'output_router_logits'):
        model.config.output_router_logits = True
    
    model.eval()
    
    # Get model config
    if hasattr(model.config, 'num_experts'):
        num_experts = model.config.num_experts
    elif hasattr(model.config, 'num_local_experts'):
        num_experts = model.config.num_local_experts
    elif hasattr(model.config, 'n_routed_experts'):
        num_experts = model.config.n_routed_experts
    else:
        raise ValueError("Cannot determine number of experts")
    
    num_layers = len(model.model.layers)
    
    # Get top-k routing parameter
    if hasattr(model.config, 'num_experts_per_tok'):
        top_k = model.config.num_experts_per_tok
    else:
        top_k = 2  # Default for Mixtral
    
    print(f"Model: {num_experts} experts, {num_layers} layers, top-{top_k} routing")
    
    # Load calibration dataset (WikiText-2)
    print("Loading WikiText-2 calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Track expert usage per layer
    expert_counts = defaultdict(lambda: defaultdict(int))
    total_tokens = defaultdict(int)
    
    # Hook to capture routing decisions from MoE gate
    routing_data = defaultdict(list)
    
    def make_routing_hook(layer_idx):
        def hook(module, input, output):
            # Debug: print what we're seeing on first layer, first call
            if layer_idx == 0 and len(routing_data[layer_idx]) == 0:
                print(f"\n[DEBUG Layer {layer_idx}] Module: {type(module).__name__}")
                print(f"[DEBUG] Output type: {type(output)}")
                if isinstance(output, tuple):
                    print(f"[DEBUG] Tuple length: {len(output)}")
                    for i, item in enumerate(output):
                        if isinstance(item, torch.Tensor):
                            print(f"[DEBUG]   Item {i}: Tensor shape {item.shape}")
                        else:
                            print(f"[DEBUG]   Item {i}: {type(item)}")
                elif isinstance(output, torch.Tensor):
                    print(f"[DEBUG] Tensor shape: {output.shape}")
            
            # Capture routing data
            if isinstance(output, torch.Tensor):
                # Router/gate output is typically logits: (batch*seq, num_experts) or (batch, seq, num_experts)
                if output.dim() == 2 and output.shape[-1] == num_experts:
                    routing_data[layer_idx].append(output.detach().cpu())
                elif output.dim() == 3 and output.shape[-1] == num_experts:
                    routing_data[layer_idx].append(output.detach().cpu())
            elif isinstance(output, tuple):
                # Try to find routing information in tuple output
                for item in output:
                    if isinstance(item, torch.Tensor):
                        # Check if this looks like routing data
                        if item.dim() == 3 and item.shape[-1] == num_experts:
                            routing_data[layer_idx].append(item.detach().cpu())
                            break
                        elif item.dim() == 2 and item.shape[-1] == num_experts:
                            routing_data[layer_idx].append(item.detach().cpu())
                            break
            
            return output
        return hook
    
    # Register hooks on MoE layers - hook the router/gate directly
    hooks = []
    print("\nInspecting layer structure...")
    moe_layers_found = 0
    
    for layer_idx, layer in enumerate(model.model.layers):
        # Check what type of MLP this layer has
        if hasattr(layer, 'mlp'):
            mlp_type = type(layer.mlp).__name__
            has_experts = hasattr(layer.mlp, 'experts')
            
            if layer_idx < 3 or has_experts:  # Debug first 3 layers or any with experts
                print(f"Layer {layer_idx}: {mlp_type}, has_experts={has_experts}")
            
            # Only hook if this is actually an MoE layer
            if has_experts:
                moe_layers_found += 1
                # Look for gate/router
                if hasattr(layer.mlp, 'gate'):
                    hook = layer.mlp.gate.register_forward_hook(make_routing_hook(layer_idx))
                    hooks.append(hook)
                elif hasattr(layer.mlp, 'router'):
                    hook = layer.mlp.router.register_forward_hook(make_routing_hook(layer_idx))
                    hooks.append(hook)
                else:
                    # Hook the whole MoE module
                    hook = layer.mlp.register_forward_hook(make_routing_hook(layer_idx))
                    hooks.append(hook)
        elif hasattr(layer, 'block_sparse_moe'):
            moe_layers_found += 1
            hook = layer.block_sparse_moe.register_forward_hook(make_routing_hook(layer_idx))
            hooks.append(hook)
    
    print(f"Found {moe_layers_found} MoE layers out of {len(model.model.layers)} total layers")
    
    if moe_layers_found == 0:
        raise ValueError("No MoE layers found! This model may not have sparse MoE architecture.")
    
    # Run inference on calibration data
    print(f"Profiling on {num_samples} samples...")
    samples_processed = 0
    
    with torch.no_grad():
        for idx, example in enumerate(tqdm(dataset, total=num_samples)):
            if samples_processed >= num_samples:
                break
            
            text = example['text']
            if len(text.strip()) < 50:  # Skip very short texts
                continue
            
            # Tokenize
            inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                             max_length=max_seq_len).to(model.device)
            
            if inputs['input_ids'].shape[1] < 10:  # Skip very short sequences
                continue
            
            # Forward pass - hooks will capture routing
            try:
                outputs = model(**inputs)
                
                # Extract routing from router_logits in output if available (Mixtral, Qwen)
                if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
                    for layer_idx, router_logits in enumerate(outputs.router_logits):
                        if router_logits is None:
                            continue
                        
                        # router_logits shape: (batch, seq_len, num_experts)
                        # Get top-k expert selections
                        batch_size, seq_len, _ = router_logits.shape
                        
                        # Get routing weights
                        routing_weights = torch.softmax(router_logits, dim=-1)
                        
                        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                        
                        # Flatten and count
                        selected_flat = selected_experts.view(-1).cpu().tolist()
                        for expert_id in selected_flat:
                            expert_counts[layer_idx][expert_id] += 1
                        
                        total_tokens[layer_idx] += batch_size * seq_len
                
                samples_processed += 1
                
                # Debug: Show routing capture status every 50 samples
                if samples_processed % 50 == 1:
                    print(f"\nSample {samples_processed}: router_logits={'YES' if hasattr(outputs, 'router_logits') and outputs.router_logits else 'NO'}, hook_data={len(routing_data)} layers")
                
            except Exception as e:
                print(f"Warning: Error processing sample: {e}")
                continue
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Process routing data captured by hooks (for DeepSeek and models without router_logits in output)
    if routing_data:
        print(f"Processing routing data from hooks ({len(routing_data)} layers)...")
        for layer_idx, routing_list in routing_data.items():
            for routing_tensor in routing_list:
                # routing_tensor could be logits, probs, or indices
                # Shape: (batch, seq, num_experts) or (batch*seq, num_experts) or (batch*seq, top_k)
                
                if routing_tensor.shape[-1] == top_k:
                    # This is already expert indices (batch*seq, top_k)
                    selected_experts = routing_tensor.long()
                    num_tokens = selected_experts.shape[0]
                else:
                    # This is logits/probs (batch*seq, num_experts) or (batch, seq, num_experts)
                    if routing_tensor.dim() == 3:
                        batch, seq, _ = routing_tensor.shape
                        routing_tensor = routing_tensor.reshape(-1, num_experts)
                        num_tokens = batch * seq
                    else:
                        num_tokens = routing_tensor.shape[0]
                    
                    # Get top-k selections
                    _, selected_experts = torch.topk(routing_tensor, top_k, dim=-1)
                
                # Count expert usage
                selected_flat = selected_experts.view(-1).tolist()
                for expert_id in selected_flat:
                    if 0 <= expert_id < num_experts:
                        expert_counts[layer_idx][expert_id] += 1
                
                total_tokens[layer_idx] += num_tokens
    
    # Build profiling JSON
    print("Building profiling data...")
    profiling_data = {
        'num_experts': num_experts,
        'layers': {}
    }
    
    for layer_idx in range(num_layers):
        if layer_idx not in expert_counts:
            continue
        
        layer_total = total_tokens[layer_idx]
        if layer_total == 0:
            continue
        
        layer_data = {
            'total_tokens': layer_total,
            'expert_stats': {}
        }
        
        for expert_id in range(num_experts):
            count = expert_counts[layer_idx].get(expert_id, 0)
            layer_data['expert_stats'][str(expert_id)] = {
                'count': count,
                'percentage': (count / layer_total * 100) if layer_total > 0 else 0,
                'activation_rate': count / layer_total if layer_total > 0 else 0
            }
        
        profiling_data['layers'][str(layer_idx)] = layer_data
    
    # Classify hot/cold based on first layer (or average across layers)
    if '0' in profiling_data['layers']:
        layer_0_stats = profiling_data['layers']['0']['expert_stats']
        
        # Adaptive threshold: hot experts = 1.5x average activation rate
        avg_activation = top_k / num_experts
        threshold = avg_activation * 1.5
        
        hot_experts = []
        cold_experts = []
        
        for expert_id, stats in layer_0_stats.items():
            if stats['activation_rate'] > threshold:
                hot_experts.append(int(expert_id))
            else:
                cold_experts.append(int(expert_id))
        
        profiling_data['hot_experts'] = hot_experts
        profiling_data['cold_experts'] = cold_experts
    else:
        profiling_data['hot_experts'] = []
        profiling_data['cold_experts'] = []
    
    # Save
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(profiling_data, f, indent=2)
    
    print(f"✓ Profiled {samples_processed} samples")
    print(f"✓ Hot experts: {len(profiling_data['hot_experts'])}, "
          f"Cold experts: {len(profiling_data['cold_experts'])}")
    
    return profiling_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='HuggingFace model name')
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Max sequence length')
    args = parser.parse_args()
    
    profile_real_routing(args.model, args.output, args.samples, args.max_seq_len)
