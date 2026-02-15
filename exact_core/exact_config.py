#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
EXACT Configuration
"""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class EXACTConfig:
    """Configuration for EXACT adaptive compression"""
    
    # Hot/Cold classification threshold
    hot_threshold: float = 0.1  # Experts activated >10% of time = hot
    
    # Sparsity patterns: (N, M, vector_length)
    hot_sparsity: Tuple[int, int, int] = (2, 4, 128)   # Light 2:4 sparsity
    cold_sparsity: Tuple[int, int, int] = (1, 4, 128)  # Aggressive 1:4 sparsity
    
    # Profiling settings
    profiling_samples: int = 1000  # Number of forward passes to profile
    
    def __repr__(self):
        return (f"EXACTConfig(hot_threshold={self.hot_threshold}, "
                f"hot={self.hot_sparsity}, cold={self.cold_sparsity})")
