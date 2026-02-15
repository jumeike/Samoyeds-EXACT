#!/usr/bin/env python3
#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
Test script for EXACT Sparsifier
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact_core import EXACTSparsifier, EXACTConfig


def test_sparsifier(profile_path: str, num_experts: int = 8):
    """Test the EXACT sparsifier"""
    
    print("="*70)
    print("EXACT Sparsifier Test")
    print("="*70)
    
    # Initialize sparsifier
    config = EXACTConfig()
    sparsifier = EXACTSparsifier(config)
    
    # Load classification from profiling results
    print(f"\n[1] Loading classification from {profile_path}")
    classification = sparsifier.load_classification(profile_path)
    
    print(f"  Hot experts ({len(classification['hot'])}): {classification['hot']}")
    print(f"  Cold experts ({len(classification['cold'])}): {classification['cold']}")
    
    # Create dummy expert weights (simulating MLP layers)
    print(f"\n[2] Creating dummy expert weights...")
    hidden_size = 4096
    intermediate_size = 14336
    
    expert_weights = []
    for i in range(num_experts):
        # Simulate gate_proj weight (hidden -> intermediate)
        weight = torch.randn(intermediate_size, hidden_size)
        expert_weights.append(weight)
    
    print(f"  Created {len(expert_weights)} expert weights")
    print(f"  Weight shape: {expert_weights[0].shape}")
    print(f"  Total params: {sum(w.numel() for w in expert_weights):,}")
    
    # Apply sparsification
    print(f"\n[3] Applying adaptive sparsification...")
    sparse_weights = sparsifier.sparsify_expert_weights(expert_weights, classification)
    
    # Compute statistics
    print(f"\n[4] Computing statistics...")
    stats = sparsifier.compute_sparsity_stats(sparse_weights, classification)
    
    print(f"\n{'='*70}")
    print(f"EXACT Sparsification Results")
    print(f"{'='*70}")
    print(f"Total experts: {stats['num_experts']}")
    print(f"Hot experts: {stats['hot_experts']} (2:4 sparsity)")
    print(f"Cold experts: {stats['cold_experts']} (1:4 sparsity)")
    print(f"\nOverall sparsity: {stats['overall_sparsity']*100:.2f}%")
    print(f"Total params: {stats['total_params']:,}")
    print(f"Non-zero params: {stats['total_nonzero']:,}")
    print(f"Memory reduction: {(1 - stats['overall_sparsity'])*100:.2f}% of dense")
    
    print(f"\nPer-Expert Breakdown:")
    print(f"{'Expert':<10} {'Type':<6} {'Sparsity':<10} {'Non-zero':<15}")
    print(f"{'-'*50}")
    for expert_id in range(num_experts):
        info = stats['per_expert'][expert_id]
        print(f"{expert_id:<10} {info['type']:<6} {info['sparsity']*100:>6.2f}%   "
              f"{info['nonzero']:>10,} / {info['params']:>10,}")
    
    # Test saving
    print(f"\n[5] Testing save functionality...")
    output_path = f"artifacts/exact_sparsification/test_sparse_weights.pt"
    sparsifier.save_sparse_weights(sparse_weights, output_path, classification)
    
    print(f"\n{'='*70}")
    print(f"✓ Test Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        profile_path = sys.argv[1]
    else:
        profile_path = "artifacts/exact_profiling/mixtral_e8.json"
    
    test_sparsifier(profile_path)
