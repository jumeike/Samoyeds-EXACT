#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
Generic Expert Profiler - Model Agnostic
"""
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from .expert_tracker import ExpertUsageTracker
from .exact_config import EXACTConfig


class ExpertProfiler:
    """Model-agnostic expert usage profiler"""
    
    def __init__(self, 
                 moe_block: torch.nn.Module,
                 num_experts: int,
                 model_name: str = "unknown",
                 config: Optional[EXACTConfig] = None):
        """
        Args:
            moe_block: The MoE block to profile
            num_experts: Number of experts in the block
            model_name: Name for logging/saving
            config: EXACT configuration
        """
        self.moe_block = moe_block
        self.num_experts = num_experts
        self.model_name = model_name
        self.config = config or EXACTConfig()
        self.tracker = ExpertUsageTracker(num_experts=num_experts, num_layers=1)
        self.bias_values = None  # Store bias for hook
        self.hook_handle = None  # Store hook handle for cleanup
        
    def apply_imbalance(self, imbalance_ratio: float = 1.5, random_pattern: bool = True):
        """
        Apply synthetic expert imbalance via gate biasing
        
        Args:
            imbalance_ratio: Strength of imbalance (0=uniform, 1.5=realistic, 3.0=extreme)
            random_pattern: Use random expert assignment (True) or deterministic descending (False)
        """
        if imbalance_ratio <= 0:
            return
            
        pattern = "random" if random_pattern else "deterministic"
        print(f"[EXACT] Applying expert imbalance (ratio={imbalance_ratio}, pattern={pattern})...")
        
        # Find the gate layer
        gate = None
        if hasattr(self.moe_block, 'gate'):
            gate = self.moe_block.gate
        elif hasattr(self.moe_block, 'router'):
            gate = self.moe_block.router
        else:
            raise ValueError("Cannot find gate/router in MoE block")
        
        # Create bias pattern for routing logits
        if random_pattern:
            torch.manual_seed(42)  # Reproducible random
            self.bias_values = torch.randn(self.num_experts) * imbalance_ratio
        else:
            self.bias_values = torch.linspace(imbalance_ratio, -imbalance_ratio, self.num_experts)
        self.bias_values = self.bias_values.to(device=gate.weight.device, dtype=gate.weight.dtype)
        
        # Use forward hook to add bias (works for all gate implementations)
        def add_bias_hook(module, input, output):
            # output can be logits or tuple (topk_idx, topk_weight, aux_loss)
            if isinstance(output, tuple):
                # DeepSeek returns (topk_idx, topk_weight, aux_loss)
                # We need to bias the logits BEFORE topk, so this won't work
                # Instead, we need a pre-hook on the linear layer
                return output
            else:
                # Mixtral/Qwen2 return logits directly
                return output + self.bias_values.unsqueeze(0).unsqueeze(0)
        
        # For DeepSeek, we need to intercept at the logits computation
        if hasattr(gate, 'scoring_func'):
            # DeepSeek-style gate: has scoring_func attribute
            # Monkey-patch to add bias
            original_forward = gate.forward
            def biased_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape
                hidden_states_flat = hidden_states.view(-1, h)
                logits = torch.nn.functional.linear(hidden_states_flat, gate.weight, self.bias_values)
                
                # Rest of DeepSeek gate logic
                if gate.scoring_func == 'softmax':
                    scores = logits.softmax(dim=-1)
                else:
                    raise NotImplementedError(f'insupportable scoring function: {gate.scoring_func}')
                
                topk_weight, topk_idx = torch.topk(scores, k=gate.top_k, dim=-1, sorted=False)
                
                if gate.top_k > 1 and gate.norm_topk_prob:
                    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
                    topk_weight = topk_weight / denominator
                
                aux_loss = None
                return topk_idx, topk_weight, aux_loss
            
            gate.forward = biased_forward
        else:
            # Mixtral/Qwen2 style - modify weight bias parameter
            if not hasattr(gate, 'bias') or gate.bias is None:
                gate.register_parameter('bias', torch.nn.Parameter(self.bias_values))
            else:
                gate.bias.data += self.bias_values
        
    def extract_routing(self, 
                       moe_output: Any,
                       router_logits: torch.Tensor,
                       top_k: int = 2) -> torch.Tensor:
        """
        Extract expert selections from MoE forward pass
        
        Args:
            moe_output: Output from MoE block (unused, for compatibility)
            router_logits: Router logits from MoE block
            top_k: Number of experts selected per token
            
        Returns:
            selected_experts: Tensor of shape (batch*seq, top_k)
        """
        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        return selected_experts
        
    def profile(self,
                batch_size: int = 1,
                seq_len: int = 512,
                hidden_size: int = 4096,
                iterations: int = 500,
                imbalance_ratio: float = 1.5,
                random_pattern: bool = True,
                top_k: int = 2) -> Dict[str, Any]:
        """
        Run profiling on the MoE block
        
        Returns:
            Dict with classification and statistics
        """
        print("="*70)
        print(f"EXACT Expert Profiling: {self.model_name}")
        print("="*70)
        print(f"Experts: {self.num_experts}, Top-K: {top_k}")
        print(f"Batch: {batch_size}, Seq len: {seq_len}, Iterations: {iterations}")
        
        # Apply imbalance
        self.apply_imbalance(imbalance_ratio, random_pattern)
        
        # Enable tracking
        self.tracker.enable()
        
        # Profiling loop
        print(f"\nProfiling...")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        with torch.no_grad():
            for i in range(iterations):
                # Random input
                hidden_states = torch.randn(batch_size, seq_len, hidden_size)
                hidden_states = hidden_states.half().cuda()
                
                # Forward pass (handle different return signatures)
                output = self.moe_block(hidden_states)
                
                # Extract routing - check if we need to get it from gate directly
                if isinstance(output, tuple):
                    moe_output, router_logits = output[0], output[1]
                    selected_experts = self.extract_routing(moe_output, router_logits, top_k)
                    self.tracker.log_routing(layer_idx=0, selected_experts=selected_experts)
                elif hasattr(self.moe_block, 'gate'):
                    # For models like DeepSeek where gate doesn't return logits
                    # We need to call gate separately to get routing info
                    gate_output = self.moe_block.gate(hidden_states)
                    if isinstance(gate_output, tuple) and len(gate_output) >= 1:
                        topk_idx = gate_output[0]  # First element is topk_idx
                        self.tracker.log_routing(layer_idx=0, selected_experts=topk_idx)
                    
                if (i + 1) % max(1, iterations // 10) == 0:
                    print(f"  Progress: {i+1}/{iterations}")
        
        # Results
        self.tracker.print_summary(layer_idx=0)
        
        # Classification
        classification = self.tracker.classify_experts(
            threshold=self.config.hot_threshold,
            layer_idx=0
        )
        
        print(f"\n{'='*70}")
        print(f"EXACT Classification (threshold={self.config.hot_threshold*100}%)")
        print(f"{'='*70}")
        print(f"Hot experts: {classification['hot']} → 2:4 sparsity")
        print(f"Cold experts: {classification['cold']} → 1:4 sparsity")
        
        # Save results with classification
        output_dir = Path('artifacts/exact_profiling')
        output_file = output_dir / f'{self.model_name}_e{self.num_experts}.json'
        self.tracker.save_stats(str(output_file), classification=classification)
        
        print(f"\n{'='*70}")
        print(f"✓ Complete! Results: {output_file}")
        print(f"{'='*70}\n")
        
        return {
            'classification': classification,
            'stats': self.tracker.get_usage_stats(layer_idx=0),
            'output_file': str(output_file)
        }
