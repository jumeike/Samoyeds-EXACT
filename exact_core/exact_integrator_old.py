#!/usr/bin/env python3
#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
EXACT Model Integrator
Applies adaptive expert-aware compression to any MoE model
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from .samoyed_converter import apply_exact_sparsification


class EXACTIntegrator:
    """
    Model-agnostic EXACT integration for MoE models
    Works with Mixtral, DeepSeek, Qwen2-MoE, etc.
    """
    
    def __init__(self, model, profiling_json: str, dense_model=None):
        """
        Args:
            model: Samoyeds MoE model (e.g., SSMixtralForCausalLM)
            profiling_json: Path to expert classification JSON
            dense_model: Optional dense model to extract original weights from
        """
        self.model = model
        self.dense_model = dense_model
        self.expert_classification = self._load_classification(profiling_json)
        
    def _load_classification(self, json_path: str) -> Dict:
        """Load expert hot/cold classification from profiling"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Format 1: Simple format {"hot_experts": [...], "cold_experts": [...]}
        if 'hot_experts' in data and 'cold_experts' in data:
            return data
        
        # Format 2: Per-layer format - return as-is, don't auto-classify
        if 'layers' in data:
            # Check if any layer has classification
            has_classifications = any(
                'classification' in layer_data 
                for layer_data in data['layers'].values()
            )
            if has_classifications:
                return data  # Use per-layer classifications directly
            else:
                # Old format without classifications - auto-classify
                return self._auto_classify_experts(data)
        
        raise ValueError(f"Unknown JSON format. Expected 'hot_experts'/'cold_experts' or 'layers' key.")
    
    def _auto_classify_experts(self, profiling_data: Dict, threshold_percentile: float = 20.0) -> Dict:
        """
        Auto-classify experts into hot/cold based on activation rates
        
        Args:
            profiling_data: Profiling output with nested layers structure
            threshold_percentile: Percentile threshold for hot/cold split (default: 20%)
        
        Returns:
            dict: {"hot_experts": [...], "cold_experts": [...]}
        """
        # Aggregate activation rates across all layers
        expert_activation = {}
        num_experts = profiling_data.get('num_experts', 8)
        
        for layer_idx, layer_data in profiling_data['layers'].items():
            expert_stats = layer_data.get('expert_stats', {})
            for expert_id, stats in expert_stats.items():
                expert_idx = int(expert_id)
                activation_rate = stats.get('activation_rate', 0.0)
                
                if expert_idx not in expert_activation:
                    expert_activation[expert_idx] = []
                expert_activation[expert_idx].append(activation_rate)
        
        # Average activation rate per expert across layers
        expert_avg_activation = {
            expert_idx: sum(rates) / len(rates) 
            for expert_idx, rates in expert_activation.items()
        }
        
        # Sort experts by activation rate
        sorted_experts = sorted(expert_avg_activation.items(), key=lambda x: x[1], reverse=True)
        
        # Split at threshold percentile
        split_idx = int(len(sorted_experts) * threshold_percentile / 100.0)
        hot_experts = [expert_idx for expert_idx, _ in sorted_experts[:split_idx]]
        cold_experts = [expert_idx for expert_idx, _ in sorted_experts[split_idx:]]
        
        print(f"\n[Auto-Classification] Threshold: {threshold_percentile}%")
        print(f"  Hot experts ({len(hot_experts)}): {hot_experts}")
        print(f"  Cold experts ({len(cold_experts)}): {cold_experts}")
        
        return {
            'hot_experts': hot_experts,
            'cold_experts': cold_experts
        }
    
    def _get_moe_layers_from_model(self, model: nn.Module) -> List[nn.Module]:
        """
        Helper: Discover all MoE layers in a given model
        Returns list of MoE blocks (e.g., SSMixtralSparseMoeBlock)
        """
        moe_layers = []
        
        # Case 1: Direct MoE block (has .experts attribute)
        if hasattr(model, 'experts'):
            moe_layers.append(model)
            return moe_layers
        
        # Case 2: Full model with model.model.layers[i] structure
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                # Look for MoE-specific attributes
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
    
    def get_moe_layers(self) -> List[nn.Module]:
        """
        Discover all MoE layers in model
        Returns list of MoE blocks (e.g., SSMixtralSparseMoeBlock)
        """
        return self._get_moe_layers_from_model(self.model)
    
    def get_expert_weights(self, expert: nn.Module, dense_expert: nn.Module = None) -> Dict[str, nn.Parameter]:
        """
        Extract weight parameters from expert module
        
        Args:
            expert: Samoyeds sparse expert (not used, for API compatibility)
            dense_expert: Dense expert module to extract original weights from
        
        Returns:
            dict: {'gate_proj': weight, 'up_proj': weight, 'down_proj': weight}
        """
        if dense_expert is None:
            raise ValueError("dense_expert required. EXACT needs original dense weights for compression.")
        
        weights = {}
        
        # Mixtral pattern: w1, w2, w3
        if hasattr(dense_expert, 'w1'):
            weights['gate_proj'] = dense_expert.w1.weight
            if hasattr(dense_expert, 'w3'):
                weights['up_proj'] = dense_expert.w3.weight
            if hasattr(dense_expert, 'w2'):
                weights['down_proj'] = dense_expert.w2.weight
                
        # DeepSeek/Qwen2 pattern: gate_proj, up_proj, down_proj
        elif hasattr(dense_expert, 'gate_proj'):
            weights['gate_proj'] = dense_expert.gate_proj.weight
            if hasattr(dense_expert, 'up_proj'):
                weights['up_proj'] = dense_expert.up_proj.weight
            if hasattr(dense_expert, 'down_proj'):
                weights['down_proj'] = dense_expert.down_proj.weight
                
        return weights
    
    # def set_expert_weights(self, expert: nn.Module, weight_name: str, 
    #                       values: torch.Tensor, indices: torch.Tensor, 
    #                       metadata: torch.Tensor, sparsity_type: str):
    #     """
    #     Replace expert weight with EXACT-sparsified Samoyeds format
        
    #     Args:
    #         expert: Expert module
    #         weight_name: 'gate_proj', 'up_proj', or 'down_proj'
    #         values, indices, metadata: Samoyeds 3-tensor format
    #         sparsity_type: 'hot' or 'cold' for N:M parameters
    #     """
    #     # Determine N:M parameters - ADAPTIVE compression with output padding
    #     if sparsity_type == 'hot':
    #         N, M = 2, 4  # Hot experts: 50% sparsity (2:4)
    #     else:  # cold
    #         N, M = 1, 4  # Cold experts: 75% sparsity (1:4)
            
    #     # Map common names to model-specific names
    #     weight_map = {
    #         'gate_proj': ['w1', 'gate_proj'],
    #         'up_proj': ['w3', 'up_proj'],
    #         'down_proj': ['w2', 'down_proj']
    #     }
        
    #     for attr_name in weight_map[weight_name]:
    #         if hasattr(expert, attr_name):
    #             layer = getattr(expert, attr_name)
                
    #             # Replace the stored sparse tensors
    #             # All three are nn.Parameters (indices and metadata have requires_grad=False)
    #             if hasattr(layer, 'weight'):
    #                 # Delete old attributes to avoid type checking issues
    #                 if hasattr(layer, 'indices'):
    #                     delattr(layer, 'indices')
    #                 if hasattr(layer, 'metadata'):
    #                     delattr(layer, 'metadata')
                    
    #                 # Set new values - all as Parameters
    #                 layer.weight = nn.Parameter(values)
    #                 layer.indices = nn.Parameter(indices, requires_grad=False)
    #                 layer.metadata = nn.Parameter(metadata, requires_grad=False)
                    
    #                 # Set N:M parameters for EXACT variable compression
    #                 layer.N = N
    #                 layer.M = M
                    
    #                 print(f"    ✓ Updated {weight_name}: {values.shape}, N={N}, M={M}")
    #             break

    def set_expert_weights(self, expert: nn.Module, weight_name: str, 
                      values: torch.Tensor, indices: torch.Tensor, 
                      metadata: torch.Tensor, sparsity_type: str):
        """
        Replace expert weight with EXACT-sparsified Samoyeds format
        Args:
            expert: Expert module
            weight_name: 'gate_proj', 'up_proj', or 'down_proj'
            values, indices, metadata: Samoyeds 3-tensor format
            sparsity_type: 'hot' or 'cold' for N:M parameters
        """
        # Store logical sparsity (for metrics/analysis)
        if sparsity_type == 'hot':
            logical_N, logical_M = 2, 4  # 50% density
        else:
            logical_N, logical_M = 1, 4  # 25% density (TRUE compression)
        
        # Hardware always uses 2:4
        hardware_N, hardware_M = 2, 4
            
        weight_map = {
            'gate_proj': ['w1', 'gate_proj'],
            'up_proj': ['w3', 'up_proj'],
            'down_proj': ['w2', 'down_proj']
        }
        
        for attr_name in weight_map[weight_name]:
            if hasattr(expert, attr_name):
                layer = getattr(expert, attr_name)
                
                if hasattr(layer, 'weight'):
                    target_device = layer.weight.device
                    values = values.to(target_device)
                    indices = indices.to(target_device)
                    metadata = metadata.to(target_device)
                    
                    if hasattr(layer, 'indices'):
                        delattr(layer, 'indices')
                    if hasattr(layer, 'metadata'):
                        delattr(layer, 'metadata')
                    
                    layer.weight = nn.Parameter(values)
                    layer.indices = nn.Parameter(indices, requires_grad=False)
                    layer.metadata = nn.Parameter(metadata, requires_grad=False)
                    
                    # Hardware format (kernel expects this)
                    layer.N = logical_N #hardware_N
                    layer.M = logical_M #hardware_M
                    
                    # Store logical sparsity for analysis
                    layer.logical_N = logical_N
                    layer.logical_M = logical_M
                    
                    # print(f"    ✓ Updated {attr_name}: {values.shape}, "
                    #     f"logical={logical_N}:{logical_M}, hardware={hardware_N}:{hardware_M}")
                break
    
    def apply_exact_to_layer(self, moe_layer: nn.Module, dense_moe_layer: nn.Module, layer_idx: int):
        """
        Apply EXACT compression to all experts in one MoE layer
        
        Args:
            moe_layer: Samoyeds sparse MoE block
            dense_moe_layer: Dense MoE block for extracting original weights
            layer_idx: Layer index for classification lookup
        """
        if not hasattr(moe_layer, 'experts'):
            print(f"  [!] Layer {layer_idx} has no experts attribute, skipping")
            return
        
        if not hasattr(dense_moe_layer, 'experts'):
            print(f"  [!] Dense layer {layer_idx} has no experts attribute, skipping")
            return
        
        num_experts = len(moe_layer.experts)
        # NEW: Get per-layer classification
        if 'layers' in self.expert_classification:
            # Per-layer format
            layer_data = self.expert_classification['layers'].get(str(layer_idx), {})
            classification = layer_data.get('classification', {})
            hot_experts = classification.get('hot', [])
            cold_experts = classification.get('cold', [])

                # DEBUG
            # if layer_idx == 0:
            #     print(f"[DEBUG] Layer 0 classification:")
            #     print(f"  Hot: {hot_experts}")
            #     print(f"  Cold: {cold_experts}")
            #     print(f"  Total classified: {len(hot_experts) + len(cold_experts)}")
        else:
            # OLD: Fallback to global classification (backward compatible)
            hot_experts = self.expert_classification.get('hot_experts', [])
            cold_experts = self.expert_classification.get('cold_experts', [])
                
            # if layer_idx == 0:
            #     print(f"[DEBUG] Using global classification:")
            #     print(f"  Hot: {hot_experts}")
            #     print(f"  Cold: {cold_experts}")
        
        # hot_experts = self.expert_classification.get('hot_experts', [])
        # cold_experts = self.expert_classification.get('cold_experts', [])
        
        # print(f"\n[Layer {layer_idx}] Processing {num_experts} experts...")
        # print(f"  Hot: {len(hot_experts)}, Cold: {len(cold_experts)}")
        
        for expert_idx in range(num_experts):
            expert = moe_layer.experts[expert_idx]
            dense_expert = dense_moe_layer.experts[expert_idx]
            
            # Determine hot/cold
            if expert_idx in hot_experts:
                sparsity_type = 'hot'
            elif expert_idx in cold_experts:
                sparsity_type = 'cold'
            else:
                print(f"  [!] Expert {expert_idx} not classified, skipping")
                continue
            
            # Get original weights from dense model
            weights = self.get_expert_weights(expert, dense_expert)
            
            if not weights:
                print(f"  [!] Expert {expert_idx} has no extractable weights")
                continue
            
            # print(f"  Expert {expert_idx} ({sparsity_type}):")
            
            # Apply EXACT sparsification to each weight with adaptive compression
            for weight_name, weight_tensor in weights.items():
                values, indices, metadata = apply_exact_sparsification(
                    weight_tensor, sparsity_type
                )
                
                # Replace in model
                self.set_expert_weights(expert, weight_name, values, indices, metadata, sparsity_type)
    
    def apply_exact_to_model(self):
        """
        Apply EXACT to entire model (all layers)
        """
        # print("="*70)
        # print("EXACT Model Integration")
        # print("="*70)
        
        if self.dense_model is None:
            raise ValueError("dense_model required for EXACT compression. Pass it during initialization.")
        
        # Get sparse and dense MoE layers
        moe_layers = self.get_moe_layers()
        # print(f"\nFound {len(moe_layers)} MoE layers")
        
        if not moe_layers:
            print("[!] No MoE layers found. Check model structure.")
            return
        
        # Get dense layers using helper method to avoid creating unnecessary integrator
        dense_moe_layers = self._get_moe_layers_from_model(self.dense_model)
        
        if len(moe_layers) != len(dense_moe_layers):
            print(f"[!] Layer count mismatch: sparse={len(moe_layers)}, dense={len(dense_moe_layers)}")
            return
        
        # Apply EXACT to each layer
        for layer_idx, (moe_layer, dense_moe_layer) in enumerate(zip(moe_layers, dense_moe_layers)):
            self.apply_exact_to_layer(moe_layer, dense_moe_layer, layer_idx)
        
        # print(f"\n{'='*70}")
        # print("✓ EXACT integration complete!")
        # print(f"{'='*70}\n")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Calculate memory footprint of EXACT-compressed model
        
        Returns:
            dict: Memory stats in MB
        """
        moe_layers = self.get_moe_layers()
        
        total_hot_memory = 0
        total_cold_memory = 0
        num_hot = 0
        num_cold = 0
        
        hot_experts = self.expert_classification.get('hot_experts', [])
        cold_experts = self.expert_classification.get('cold_experts', [])
        
        for moe_layer in moe_layers:
            if not hasattr(moe_layer, 'experts'):
                continue
                
            for expert_idx, expert in enumerate(moe_layer.experts):
                weights = self.get_expert_weights(expert)
                
                for weight_tensor in weights.values():
                    memory = weight_tensor.numel() * weight_tensor.element_size()
                    
                    if expert_idx in hot_experts:
                        total_hot_memory += memory
                        num_hot += 1
                    elif expert_idx in cold_experts:
                        total_cold_memory += memory
                        num_cold += 1
        
        total_memory_mb = (total_hot_memory + total_cold_memory) / 1024**2
        hot_memory_mb = total_hot_memory / 1024**2
        cold_memory_mb = total_cold_memory / 1024**2
        
        return {
            'total_mb': total_memory_mb,
            'hot_mb': hot_memory_mb,
            'cold_mb': cold_memory_mb,
            'num_hot_experts': num_hot,
            'num_cold_experts': num_cold
        }


def apply_exact_to_model(model, profiling_json: str, model_name_or_path: str = None, dense_model = None):
    """
    Convenience function: Apply EXACT to model in one call
    
    Args:
        model: Samoyeds MoE model
        profiling_json: Path to expert classification JSON
        model_name_or_path: HF checkpoint path to load dense weights from
        dense_model: Optional pre-loaded dense model (if None, will load from checkpoint)
        
    Returns:
        model: Modified model with EXACT compression
    """
    # Load dense model if not provided
    if dense_model is None:
        if model_name_or_path is None:
            raise ValueError("Either dense_model or model_name_or_path must be provided")
        
        print(f"Loading dense model from {model_name_or_path}...")
        from transformers import AutoModelForCausalLM
        dense_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map='cpu'  # Load to CPU to save GPU memory
        )
        print("✓ Dense model loaded")
    
    integrator = EXACTIntegrator(model, profiling_json, dense_model=dense_model)
    integrator.apply_exact_to_model()
    return model
