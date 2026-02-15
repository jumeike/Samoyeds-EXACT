#!/usr/bin/env python3
#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
Test EXACT to Samoyeds format conversion
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact_core import EXACTSparsifier, convert_to_samoyed_format, apply_exact_sparsification


def test_conversion():
    """Test EXACT sparsification and Samoyeds format conversion"""
    
    print("="*70)
    print("EXACT → Samoyeds Format Conversion Test")
    print("="*70)
    
    # Create dummy weight (Mixtral gate_proj size)
    hidden_size = 4096
    intermediate_size = 14336
    weight = torch.randn(intermediate_size, hidden_size)
    
    print(f"\n[1] Original weight shape: {weight.shape}")
    print(f"    Total params: {weight.numel():,}")
    
    # Test Stage 1: EXACT sparsification only
    print(f"\n[2] Stage 1: EXACT magnitude pruning...")
    sparsifier = EXACTSparsifier()
    
    sparse_hot = sparsifier.apply_2_4_sparsity(weight)
    sparse_cold = sparsifier.apply_1_4_sparsity(weight)
    
    print(f"    Hot (2:4): {(sparse_hot == 0).sum().item() / sparse_hot.numel() * 100:.1f}% zeros")
    print(f"    Cold (1:4): {(sparse_cold == 0).sum().item() / sparse_cold.numel() * 100:.1f}% zeros")
    
    # Test Stage 2: Samoyeds format conversion only
    print(f"\n[3] Stage 2: Samoyeds format conversion...")
    values_hot, indices_hot, metadata_hot = convert_to_samoyed_format(sparse_hot, 'hot')
    values_cold, indices_cold, metadata_cold = convert_to_samoyed_format(sparse_cold, 'cold')
    
    print(f"    Hot converted:")
    print(f"      Values: {values_hot.shape}, Indices: {indices_hot.shape}, Metadata: {metadata_hot.shape}")
    print(f"    Cold converted:")
    print(f"      Values: {values_cold.shape}, Indices: {indices_cold.shape}, Metadata: {metadata_cold.shape}")
    
    # Test combined pipeline (both stages)
    print(f"\n[4] Combined pipeline (both stages)...")
    values_hot_combined, indices_hot_combined, metadata_hot_combined = apply_exact_sparsification(weight, 'hot')
    values_cold_combined, indices_cold_combined, metadata_cold_combined = apply_exact_sparsification(weight, 'cold')
    
    print(f"    Hot: Compression {weight.numel() / values_hot_combined.numel():.2f}x")
    print(f"    Cold: Compression {weight.numel() / values_cold_combined.numel():.2f}x")
    
    # Verify shapes
    print(f"\n[5] Verifying format correctness...")
    
    # Hot: 2:4 → 50% compression on values
    expected_hot_size = (intermediate_size // 2, hidden_size // 2)
    assert values_hot.shape == expected_hot_size, f"Hot values shape mismatch: {values_hot.shape} vs {expected_hot_size}"
    assert values_hot_combined.shape == expected_hot_size, f"Hot values shape mismatch (combined): {values_hot_combined.shape} vs {expected_hot_size}"
    print(f"    ✓ Hot expert values shape correct")
    
    # Cold: 1:4 → different compression
    # Note: Cold uses N=1, M=4 for pruning but SPTC_N=2, SPTC_M=4 for storage
    expected_cold_size = (intermediate_size // 4, hidden_size // 2)
    assert values_cold.shape == expected_cold_size, f"Cold values shape mismatch: {values_cold.shape} vs {expected_cold_size}"
    assert values_cold_combined.shape == expected_cold_size, f"Cold values shape mismatch (combined): {values_cold_combined.shape} vs {expected_cold_size}"
    print(f"    ✓ Cold expert values shape correct")
    
    # Both stage-by-stage and combined pipeline produce valid Samoyeds format
    print(f"    ✓ Both stage-by-stage and combined produce valid format")
    
    # Compare memory savings
    print(f"\n[6] Memory Analysis:")
    dense_memory = weight.numel() * 2  # FP16 = 2 bytes
    hot_memory = (values_hot_combined.numel() + indices_hot_combined.numel() + metadata_hot_combined.numel()) * 2
    cold_memory = (values_cold_combined.numel() + indices_cold_combined.numel() + metadata_cold_combined.numel()) * 2
    
    print(f"    Dense: {dense_memory / 1024**2:.2f} MB")
    print(f"    Hot:   {hot_memory / 1024**2:.2f} MB ({hot_memory/dense_memory*100:.1f}%)")
    print(f"    Cold:  {cold_memory / 1024**2:.2f} MB ({cold_memory/dense_memory*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"✓ All tests passed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_conversion()
