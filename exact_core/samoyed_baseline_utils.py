#!/usr/bin/env python3
#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
Samoyed Baseline Utilities
Uniform 2:4 magnitude pruning for baseline comparisons
"""

import torch
import torch.nn as nn
from typing import List
from .samoyed_converter import apply_exact_sparsification


def get_moe_layers_from_model(model: nn.Module) -> List[nn.Module]:
    """
    Discover all MoE layers in a model
    Returns list of MoE blocks (e.g., SSMixtralSparseMoeBlock)
    """
    moe_layers = []
    
    # Case 1: Direct MoE block (has .experts attribute)
    if hasattr(model, 'experts'):
        moe_layers.append(model)
        return moe_layers
    
    # Case 1.5: Single decoder layer (has .mlp.experts)
    if hasattr(model, 'mlp') and hasattr(model.mlp, 'experts'):
        moe_layers.append(model.mlp)
        return moe_layers
    
    # Case 2: Full model with model.model.layers[i] structure
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            if hasattr(layer, 'block_sparse_moe'):
                moe_layers.append(layer.block_sparse_moe)
            elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                moe_layers.append(layer.mlp)
    
    # Case 3: Model.layers directly (some models)
    elif hasattr(model, 'layers'):
        for layer in model.layers:
            if hasattr(layer, 'block_sparse_moe'):
                moe_layers.append(layer.block_sparse_moe)
            elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                moe_layers.append(layer.mlp)
    
    return moe_layers


def apply_uniform_sparsification(sparse_model: nn.Module, dense_model: nn.Module):
    """
    Apply uniform 2:4 magnitude pruning to all experts in MoE model
    
    Args:
        sparse_model: Samoyeds sparse MoE model (with sparse layer wrappers)
        dense_model: Dense MoE model (source of original weights)
    """
    print("\nApplying uniform 2:4 magnitude pruning to all experts...")
    
    sparse_moe_layers = get_moe_layers_from_model(sparse_model)
    dense_moe_layers = get_moe_layers_from_model(dense_model)
    
    if len(sparse_moe_layers) != len(dense_moe_layers):
        raise ValueError(f"Layer mismatch: sparse={len(sparse_moe_layers)}, dense={len(dense_moe_layers)}")
    
    total_experts = 0
    
    for layer_idx, (sparse_layer, dense_layer) in enumerate(zip(sparse_moe_layers, dense_moe_layers)):
        if not hasattr(sparse_layer, 'experts') or not hasattr(dense_layer, 'experts'):
            continue
        
        num_experts = len(sparse_layer.experts)
        
        for expert_idx in range(num_experts):
            sparse_expert = sparse_layer.experts[expert_idx]
            dense_expert = dense_layer.experts[expert_idx]
            
            # Determine weight attribute names (Mixtral vs DeepSeek/Qwen)
            if hasattr(dense_expert, 'w1'):
                weight_names = [('w1', 'w1'), ('w3', 'w3'), ('w2', 'w2')]
            elif hasattr(dense_expert, 'gate_proj'):
                weight_names = [('gate_proj', 'gate_proj'), ('up_proj', 'up_proj'), ('down_proj', 'down_proj')]
            else:
                continue
            
            # Apply uniform 2:4 magnitude pruning to each weight and convert to sparse format
            for sparse_attr, dense_attr in weight_names:
                if not hasattr(dense_expert, dense_attr) or not hasattr(sparse_expert, sparse_attr):
                    continue
                
                dense_weight = getattr(dense_expert, dense_attr).weight
                values, indices, metadata = apply_exact_sparsification(dense_weight, sparsity_type='hot')
                
                sparse_layer_obj = getattr(sparse_expert, sparse_attr)
                
                if hasattr(sparse_layer_obj, 'weight'):
                    target_device = sparse_layer_obj.weight.device
                    values = values.to(target_device)
                    indices = indices.to(target_device)
                    metadata = metadata.to(target_device)
                    
                    # Set sparse format (values, indices, metadata)
                    sparse_layer_obj.weight = nn.Parameter(values)
                    sparse_layer_obj.indices = nn.Parameter(indices, requires_grad=False)
                    sparse_layer_obj.metadata = nn.Parameter(metadata, requires_grad=False)
                    sparse_layer_obj.N = 2
                    sparse_layer_obj.M = 4
            
            total_experts += 1
    
    print(f"✓ Applied uniform 2:4 pruning to {total_experts} experts across {len(sparse_moe_layers)} layers")


def apply_uniform_sparsification_single_model(model: nn.Module):
    """
    Apply uniform 2:4 magnitude pruning to model without separate dense source
    Uses existing weights in model (for benchmarking scripts)
    
    Args:
        model: Samoyeds sparse MoE model with initialized weights
    """
    print("\nApplying uniform 2:4 magnitude pruning to all experts...")
    
    moe_layers = get_moe_layers_from_model(model)
    total_experts = 0
    
    for layer_idx, moe_layer in enumerate(moe_layers):
        if not hasattr(moe_layer, 'experts'):
            continue
        
        num_experts = len(moe_layer.experts)
        
        for expert_idx in range(num_experts):
            expert = moe_layer.experts[expert_idx]
            
            # Determine weight attribute names
            if hasattr(expert, 'w1'):
                weight_attrs = ['w1', 'w3', 'w2']
            elif hasattr(expert, 'gate_proj'):
                weight_attrs = ['gate_proj', 'up_proj', 'down_proj']
            else:
                continue
            
            # Apply uniform 2:4 pruning to each weight
            for attr_name in weight_attrs:
                if not hasattr(expert, attr_name):
                    continue
                
                layer_obj = getattr(expert, attr_name)
                
                if hasattr(layer_obj, 'weight'):
                    # Get current weight as source
                    dense_weight = layer_obj.weight.data.clone()
                    values, indices, metadata = apply_exact_sparsification(dense_weight, sparsity_type='hot')
                    
                    target_device = layer_obj.weight.device
                    values = values.to(target_device)
                    indices = indices.to(target_device)
                    metadata = metadata.to(target_device)
                    
                    # Clean up old attributes
                    if hasattr(layer_obj, 'indices'):
                        delattr(layer_obj, 'indices')
                    if hasattr(layer_obj, 'metadata'):
                        delattr(layer_obj, 'metadata')
                    
                    # Set sparse format
                    layer_obj.weight = nn.Parameter(values)
                    layer_obj.indices = nn.Parameter(indices, requires_grad=False)
                    layer_obj.metadata = nn.Parameter(metadata, requires_grad=False)
                    layer_obj.N = 2
                    layer_obj.M = 4
            
            total_experts += 1
    
    print(f"✓ Applied uniform 2:4 pruning to {total_experts} experts across {len(moe_layers)} layers")
