#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
EXACT-Compressed DeepSeek Model
Applies adaptive expert-aware compression on top of Samoyeds sparse execution
"""

import sys
from pathlib import Path

# Add parent directory for exact_core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepseek.modeling_deepseek import DeepseekMoE
from deepseek.modeling_deepseek_samoyeds import SSDeepseekMoE
from exact_core import apply_exact_to_model


def deepseek_moe_to_ss(original: DeepseekMoE) -> SSDeepseekMoE:
    """Convert dense DeepSeek MoE to Samoyeds format"""
    # Check by class name to handle different import paths
    if type(original).__name__ == 'DeepseekMoE':
        # DeepseekMoE has .config attribute, use it directly
        from deepseek.modeling_deepseek_samoyeds import SSDeepseekMoE
        ss_block = SSDeepseekMoE(original.config, skip_sparsifier=True)
        # Copy gate weights
        ss_block.gate.weight.data = original.gate.weight.data.clone()
        return ss_block
    return original


def deepseek_moe_to_exact(original: DeepseekMoE, profiling_json: str, model_name_or_path: str = None) -> SSDeepseekMoE:
    """
    Convert dense DeepSeek MoE to EXACT-compressed version
    
    EXACT applies adaptive per-expert compression based on usage profiling:
    - Hot experts: 2:4 sparsity (50% density)
    - Cold experts: 1:4 sparsity (25% density)
    
    NOTE: DeepSeek has both shared and routed experts. EXACT only compresses
    routed experts, keeping shared experts at uniform 2:4 sparsity.
    
    Args:
        original: Dense DeepseekMoE block
        profiling_json: Path to expert classification JSON
            Format: {"hot_experts": [0,1,2], "cold_experts": [3,4,5]}
        model_name_or_path: Optional HF checkpoint path (not needed for block conversion)
    
    Returns:
        SSDeepseekMoE with EXACT adaptive compression applied
    
    Example:
        >>> dense_moe = DeepseekMoE(config)
        >>> exact_moe = deepseek_moe_to_exact(dense_moe, "artifacts/exact_profiling/deepseek_e160.json")
    """
    # Save reference to original dense block
    dense_block = original
    
    # Step 1: Convert to Samoyeds format (uniform sparse structure)
    samoyeds_block = deepseek_moe_to_ss(original)
    
    # Step 2: Apply EXACT adaptive compression using ORIGINAL dense weights
    exact_block = apply_exact_to_model(samoyeds_block, profiling_json, 
                                       model_name_or_path=None, 
                                       dense_model=dense_block)
    
    return exact_block
