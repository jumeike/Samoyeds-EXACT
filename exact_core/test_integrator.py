#!/usr/bin/env python3
#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
Test EXACT model integration with dummy MoE structure
"""

import torch
import torch.nn as nn
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact_core import EXACTIntegrator
from module.linear.SSFusedSiluTransLinear import SSFusedSiluTransLinear
from module.linear.SSTransLinear import SSTransLinear
from module.linear.SPDenseWeightedTransLinear import SPDenseWeightedLinear


class DummyExpert(nn.Module):
    """Mimics Samoyeds expert structure"""
    def __init__(self, hidden_size=4096, intermediate_size=14336):
        super().__init__()
        # Mixtral/DeepSeek naming: w1, w2, w3
        self.w1 = SSFusedSiluTransLinear(nn.Linear(hidden_size, intermediate_size, bias=False))
        self.w3 = SSTransLinear(nn.Linear(hidden_size, intermediate_size, bias=False))
        self.w2 = SPDenseWeightedLinear(nn.Linear(intermediate_size, hidden_size, bias=False))


class DummyMoELayer(nn.Module):
    """Mimics Samoyeds MoE block"""
    def __init__(self, num_experts=8):
        super().__init__()
        self.experts = nn.ModuleList([DummyExpert() for _ in range(num_experts)])


class DummyLayer(nn.Module):
    """Mimics transformer layer with MoE"""
    def __init__(self, num_experts=8):
        super().__init__()
        self.block_sparse_moe = DummyMoELayer(num_experts)


class DummyModelInner(nn.Module):
    """Inner model structure"""
    def __init__(self, num_layers=2, num_experts=8):
        super().__init__()
        self.layers = nn.ModuleList([DummyLayer(num_experts) for _ in range(num_layers)])


class DummyModel(nn.Module):
    """Mimics full Samoyeds model structure"""
    def __init__(self, num_layers=2, num_experts=8):
        super().__init__()
        self.model = DummyModelInner(num_layers, num_experts)


def test_integrator():
    """Test EXACT integration pipeline"""
    
    print("="*70)
    print("EXACT Integrator Test (Dummy Model)")
    print("="*70)
    
    # Create dummy profiling data
    profiling_path = Path(__file__).parent / "test_profiling.json"
    profiling_data = {
        "hot_experts": [0, 1, 2, 3],  # First 4 are hot
        "cold_experts": [4, 5, 6, 7]  # Last 4 are cold
    }
    
    with open(profiling_path, 'w') as f:
        json.dump(profiling_data, f)
    
    print(f"\n[1] Created test profiling: {profiling_path}")
    print(f"    Hot: {profiling_data['hot_experts']}")
    print(f"    Cold: {profiling_data['cold_experts']}")
    
    # Create dummy model
    model = DummyModel(num_layers=2, num_experts=8)
    print(f"\n[2] Created dummy model:")
    print(f"    Layers: 2")
    print(f"    Experts per layer: 8")
    print(f"    Total experts: 16")
    
    # Initialize integrator
    integrator = EXACTIntegrator(model, str(profiling_path))
    print(f"\n[3] Initialized EXACT integrator")
    
    # Test layer discovery
    moe_layers = integrator.get_moe_layers()
    print(f"\n[4] Discovered {len(moe_layers)} MoE layers")
    assert len(moe_layers) == 2, f"Expected 2 layers, found {len(moe_layers)}"
    
    # Test weight extraction
    print(f"\n[5] Testing weight extraction...")
    expert_0 = moe_layers[0].experts[0]
    weights = integrator.get_expert_weights(expert_0)
    print(f"    Extracted weights: {list(weights.keys())}")
    assert 'gate_proj' in weights, "Missing gate_proj"
    assert 'up_proj' in weights, "Missing up_proj"
    assert 'down_proj' in weights, "Missing down_proj"
    print(f"    ✓ All weights present")
    
    # Test full integration (this will actually sparsify)
    print(f"\n[6] Applying EXACT to model...")
    integrator.apply_exact_to_model()
    
    # Get memory stats
    print(f"\n[7] Memory statistics:")
    stats = integrator.get_memory_stats()
    print(f"    Total: {stats['total_mb']:.2f} MB")
    print(f"    Hot: {stats['hot_mb']:.2f} MB ({stats['num_hot_experts']} expert weights)")
    print(f"    Cold: {stats['cold_mb']:.2f} MB ({stats['num_cold_experts']} expert weights)")
    
    # Cleanup
    profiling_path.unlink()
    print(f"\n[8] Cleaned up test files")
    
    print(f"\n{'='*70}")
    print("✓ All integration tests passed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_integrator()
