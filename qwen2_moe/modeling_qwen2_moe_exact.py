#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
EXACT-Compressed Qwen2-MoE Model
Applies adaptive expert-aware compression on top of Samoyeds sparse execution
"""

import sys
from pathlib import Path

# Add parent directory for exact_core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from qwen2_moe.modeling_qwen2_moe_samoyeds import SSQwen2MoeSparseMoeBlock
from exact_core import apply_exact_to_model


def qwen2_moe_to_ss(original: Qwen2MoeSparseMoeBlock) -> SSQwen2MoeSparseMoeBlock:
    """Convert dense Qwen2-MoE to Samoyeds format"""
    # Check by class name to handle different import paths
    if type(original).__name__ == 'Qwen2MoeSparseMoeBlock':
        # Create a minimal config from the original block's attributes
        from qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
        config = Qwen2MoeConfig(
            num_experts=original.num_experts,
            num_experts_per_tok=original.top_k,
            norm_topk_prob=original.norm_topk_prob,
            hidden_size=original.gate.in_features,
            moe_intermediate_size=original.experts[0].gate_proj.out_features,
            shared_expert_intermediate_size=original.shared_expert.gate_proj.out_features,
        )
        ss_block = SSQwen2MoeSparseMoeBlock(config, skip_sparsifier=True)
        # Copy weights from original
        ss_block.gate.weight.data = original.gate.weight.data.clone()
        ss_block.shared_expert_gate.weight.data = original.shared_expert_gate.weight.data.clone()
        return ss_block
    return original


def qwen2_moe_to_exact(original: Qwen2MoeSparseMoeBlock, profiling_json: str, model_name_or_path: str = None) -> SSQwen2MoeSparseMoeBlock:
    """
    Convert dense Qwen2-MoE to EXACT-compressed version
    
    EXACT applies adaptive per-expert compression based on usage profiling:
    - Hot experts: 2:4 sparsity (50% density)
    - Cold experts: 1:4 sparsity (25% density)
    
    NOTE: Qwen2-MoE has both shared and routed experts. EXACT only compresses
    routed experts, keeping shared expert at uniform 2:4 sparsity.
    
    Args:
        original: Dense Qwen2MoeSparseMoeBlock
        profiling_json: Path to expert classification JSON
            Format: {"hot_experts": [0,1,2], "cold_experts": [3,4,5]}
        model_name_or_path: Optional HF checkpoint path (not needed for block conversion)
    
    Returns:
        SSQwen2MoeSparseMoeBlock with EXACT adaptive compression applied
    
    Example:
        >>> dense_moe = Qwen2MoeSparseMoeBlock(config)
        >>> exact_moe = qwen2_moe_to_exact(dense_moe, "artifacts/exact_profiling/qwen2_moe_e60.json")
    """
    # Save reference to original dense block
    dense_block = original
    
    # Step 1: Convert to Samoyeds format (uniform sparse structure)
    samoyeds_block = qwen2_moe_to_ss(original)
    
    # Step 2: Apply EXACT adaptive compression using ORIGINAL dense weights
    exact_block = apply_exact_to_model(samoyeds_block, profiling_json, 
                                       model_name_or_path=None, 
                                       dense_model=dense_block)
    
    return exact_block
