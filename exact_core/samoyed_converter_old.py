#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
EXACT to Samoyeds Format Converter
Bridges EXACT's adaptive sparsification to Samoyeds execution format
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import samoyeds_kernel
except ImportError:
    print("[EXACT] Warning: samoyeds_kernel not available. Install Samoyeds first.")
    samoyeds_kernel = None


def convert_to_samoyed_format(sparse_weight: torch.Tensor, 
                              sparsity_type: str = 'hot',
                              vector_length: int = 128) -> tuple:
    """
    Convert EXACT-sparsified weight to Samoyeds 3-tensor format
    
    Args:
        sparse_weight: PyTorch tensor with N:M sparsity pattern (has zeros)
        sparsity_type: 'hot' (2:4) or 'cold' (1:4)
        vector_length: Vector length for Sub-Row grouping (default: 128)
        
    Returns:
        (values, indices, metadata): Samoyeds format tensors
    """
    if samoyeds_kernel is None:
        raise RuntimeError("samoyeds_kernel not available. Please install Samoyeds.")
    
    # Determine N:M parameters - ADAPTIVE compression
    if sparsity_type == 'hot':
        N, M = 2, 4  # Hot experts: 50% sparsity (2:4)
    elif sparsity_type == 'cold':
        N, M = 1, 4  # Cold experts: 75% sparsity (1:4)
    else:
        raise ValueError(f"Unknown sparsity_type: {sparsity_type}. Use 'hot' or 'cold'.")
    
    # EXACT Dual-Storage: Allocate based on ACTUAL N:M (not fixed 2:4)
    # Storage format matches input sparsity, kernel expands during load
    nrows, ncols = sparse_weight.shape
    NUM_OF_META_PER_UINT = 16
    
    # N:M sparsity compresses COLUMNS only (keep top-N per M consecutive values in each row)
    # - 2:4 (hot): nrows × ncols/2 values (Samoyeds baseline)
    # - 1:4 (cold): nrows × ncols/4 values (EXACT adaptive storage)
    values_rows = nrows  # Rows unchanged
    values_cols = ncols // M * N  # Columns compressed: ncols/2 for 2:4, ncols/4 for 1:4
    
    A_values = torch.zeros((values_rows, values_cols), 
                          dtype=sparse_weight.dtype, device='cpu')
    A_indices = torch.zeros((ncols // vector_length, values_rows), 
                           dtype=torch.int32, device='cpu')
    A_metadata = torch.zeros((values_rows, values_cols // NUM_OF_META_PER_UINT), 
                            dtype=torch.int32, device='cpu')
    
    # Convert to CPU for kernel processing
    weight_cpu = sparse_weight.cpu().detach().clone()
    
    # Ensure tensor is contiguous and float16 (Samoyeds format)
    if not weight_cpu.is_contiguous():
        weight_cpu = weight_cpu.contiguous()
    if weight_cpu.dtype != torch.float16:
        weight_cpu = weight_cpu.half()
    
    # Update output tensor dtype to match
    A_values = A_values.half()
    
    # Call Samoyeds sparsifier with actual N:M parameters
    # - For 2:4 (hot): Standard Samoyeds path (backward compatible)
    # - For 1:4 (cold): Compact storage, kernel expands during GM→SM load
    samoyeds_kernel.sparsifier(
        weight_cpu,
        A_values,
        A_indices,
        A_metadata,
        nrows, ncols, vector_length, N, M
    )
    
    # Move back to original device
    device = sparse_weight.device
    A_values = A_values.to(device)
    A_indices = A_indices.to(device)
    A_metadata = A_metadata.to(device)

    # print(f"[DEBUG] sparsity_type={sparsity_type}, N={N}, M={M}")
    # print(f"[DEBUG] values.shape={A_values.shape}, expected=({nrows//M*N}, {ncols//M*N})")
    # print(f"[DEBUG] indices.shape={A_indices.shape}, metadata.shape={A_metadata.shape}")
    
    return A_values, A_indices, A_metadata


# def convert_to_samoyed_format(sparse_weight: torch.Tensor, 
#                               sparsity_type: str = 'hot',
#                               vector_length: int = 128) -> tuple:
#     """
#     Convert to Samoyeds 3-tensor format
#     NOTE: Always uses N=2, M=4 for hardware compatibility

#     Args:
#         sparse_weight: PyTorch tensor with N:M sparsity pattern (has zeros)
#         sparsity_type: 'hot' (2:4) or 'cold' (2:4) - unused. Always 2:4 for hardware
#         vector_length: Vector length for Sub-Row grouping (default: 128)
#     """
#     if samoyeds_kernel is None:
#         raise RuntimeError("samoyeds_kernel not available.")
    
#     # ALWAYS use 2:4 for hardware (even for pre-pruned 1:4 weights)
#     N, M = 2, 4
#     SPTC_N, SPTC_M = 2, 4
#     nrows, ncols = sparse_weight.shape
#     NUM_OF_META_PER_UINT = 16
    
#     values_rows = nrows // SPTC_M * SPTC_N
#     values_cols = ncols // SPTC_M * SPTC_N
    
#     A_values = torch.zeros((values_rows, values_cols), 
#                           dtype=torch.float16, device='cpu')
#     A_indices = torch.zeros((ncols // vector_length, values_rows), 
#                            dtype=torch.int32, device='cpu')
#     A_metadata = torch.zeros((values_rows, values_cols // NUM_OF_META_PER_UINT), 
#                             dtype=torch.int32, device='cpu')
    
#     weight_cpu = sparse_weight.cpu().detach().clone()
#     if not weight_cpu.is_contiguous():
#         weight_cpu = weight_cpu.contiguous()
#     if weight_cpu.dtype != torch.float16:
#         weight_cpu = weight_cpu.half()
    
#     A_values = A_values.half()
    
#     samoyeds_kernel.sparsifier(
#         weight_cpu,
#         A_values,
#         A_indices,
#         A_metadata,
#         nrows, ncols, vector_length, N, M
#     )
    
#     return A_values, A_indices, A_metadata

def apply_exact_sparsification(weight: torch.Tensor, 
                               sparsity_type: str = 'hot') -> tuple:
    """
    Complete EXACT sparsification pipeline with TRUE adaptive compression
    - Hot experts: 2:4 pruning (50% density)
    - Cold experts: 1:4 pruning (25% density) then padded to 2:4 format

    Args:
        weight: Dense weight tensor
        sparsity_type: 'hot' (2:4) or 'cold' (1:4)
    Returns:
        (values, indices, metadata): Samoyeds format ready for execution
    """
    nrows, ncols = weight.shape
    # weight_pruned = weight.clone()
    # Convert to FP32 for CPU operations
    weight_pruned = weight.float().clone()
    
    if sparsity_type == 'cold':
        # 1:4 pruning: keep top-1 per group of 4
        reshaped = weight_pruned.view(nrows, ncols // 4, 4)
        abs_vals = reshaped.abs()
        top1_idx = abs_vals.argmax(dim=2, keepdim=True)
        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(2, top1_idx, True)
        reshaped = torch.where(mask, reshaped, torch.zeros_like(reshaped))
        weight_pruned = reshaped.view(nrows, ncols)
    else:
        # HOT: 2:4 pruning - keep top-2 per group of 4
        reshaped = weight_pruned.view(nrows, ncols // 4, 4)
        abs_vals = reshaped.abs()
        _, top2_idx = abs_vals.topk(2, dim=2)
        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(2, top2_idx, True)
        reshaped = torch.where(mask, reshaped, torch.zeros_like(reshaped))
        weight_pruned = reshaped.view(nrows, ncols)
    
    # Convert back to FP16 for packing
    weight_pruned = weight_pruned.half()
    
    # Pack as 2:4 format for hardware
    values, indices, metadata = convert_to_samoyed_format(weight_pruned, sparsity_type=sparsity_type)
    return values, indices, metadata

# def apply_exact_sparsification(weight: torch.Tensor, 
#                                sparsity_type: str = 'hot') -> tuple:
#     """
#     Complete EXACT sparsification pipeline:
#     1. Pass dense weight directly to Samoyeds packer (it does the selection)
    
#     NOTE: The Samoyeds packer expects DENSE input and extracts sparse values
#     based on N:M parameters. DO NOT pre-prune the weights!
    
#     Args:
#         weight: Dense weight tensor (NOT pre-pruned)
#         sparsity_type: 'hot' (2:4) or 'cold' (1:4)
        
#     Returns:
#         (values, indices, metadata): Samoyeds format ready for execution
#     """
#     # Pass dense weight directly to packer - it will extract based on N:M
#     values, indices, metadata = convert_to_samoyed_format(weight, sparsity_type)
    
#     return values, indices, metadata


def verify_sparsity_pattern(weight: torch.Tensor, sparsity_type: str) -> bool:
    """
    Verify that weight tensor has correct N:M sparsity pattern
    
    Args:
        weight: Weight tensor to verify
        sparsity_type: 'hot' (2:4) or 'cold' (1:4)
        
    Returns:
        True if pattern is correct, False otherwise
    """
    if sparsity_type == 'hot':
        N, M = 2, 4
    elif sparsity_type == 'cold':
        N, M = 1, 4
    else:
        return False
    
    out_features, in_features = weight.shape
    if in_features % M != 0:
        return False
    
    reshaped = weight.view(out_features, in_features // M, M)
    nonzero_per_group = (reshaped != 0).sum(dim=2)
    
    # Check all groups have exactly N non-zeros
    return (nonzero_per_group == N).all().item()
