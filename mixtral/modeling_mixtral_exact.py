#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
EXACT-Compressed Mixtral Model
Applies adaptive expert-aware compression on top of Samoyeds sparse execution
"""

import sys
from pathlib import Path
import json

# Add parent directory for exact_core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mixtral.modeling_mixtral import MixtralSparseMoeBlock
from mixtral.modeling_mixtral_samoyeds import SSMixtralSparseMoeBlock, sparsemoeblock_to_ss
from exact_core import apply_exact_to_model


def sparsemoeblock_to_exact(original: MixtralSparseMoeBlock, profiling_json: str, model_name_or_path: str = None) -> SSMixtralSparseMoeBlock:
    """
    Convert dense Mixtral MoE block to EXACT-compressed version
    
    EXACT applies adaptive per-expert compression based on usage profiling:
    - Hot experts: 2:4 sparsity (50% density)
    - Cold experts: 1:4 sparsity (25% density)
    
    Args:
        original: Dense MixtralSparseMoeBlock
        profiling_json: Path to expert classification JSON
            Format: {"hot_experts": [0,1,2], "cold_experts": [3,4,5]}
        model_name_or_path: Optional HF checkpoint path (not needed for block conversion)
    
    Returns:
        SSMixtralSparseMoeBlock with EXACT adaptive compression applied
    
    Example:
        >>> dense_moe = MixtralSparseMoeBlock(config)
        >>> exact_moe = sparsemoeblock_to_exact(dense_moe, "artifacts/exact_profiling/mixtral_e8.json")
    """
    # CRITICAL: Save reference to original dense block BEFORE any conversion
    dense_block = original
    
    # Step 1: Convert to Samoyeds format (uniform sparse structure)
    samoyeds_block = sparsemoeblock_to_ss(original, skip_sparsifier=True)
    
    # Step 2: Apply EXACT adaptive compression using ORIGINAL dense weights
    exact_block = apply_exact_to_model(samoyeds_block, profiling_json, 
                                       model_name_or_path=None, 
                                       dense_model=dense_block)
    
    return exact_block

