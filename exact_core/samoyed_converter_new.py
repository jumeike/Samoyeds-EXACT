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
    
    # C++ sparsifier ONLY supports 2:4 (hardware constraint)
    # Logical sparsity (1:4 vs 2:4) is tracked at Python layer (set_expert_weights)
    N, M = 2, 4
    
    nrows, ncols = sparse_weight.shape
    NUM_OF_META_PER_UINT = 16
    
    # Always allocate for 2:4 format (C++ sparsifier requirement)
    values_rows = nrows // M * N  # Always nrows/2
    values_cols = ncols // M * N  # Always ncols/2
    
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
    
    # Always call with 2:4 (C++ sparsifier limitation)
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
    
    return A_values, A_indices, A_metadata

# def apply_exact_sparsification(weight: torch.Tensor, 
#                                sparsity_type: str = 'hot') -> tuple:
#     """
#     Complete EXACT sparsification pipeline with TRUE adaptive compression
#     - Hot experts: 2:4 pruning (50% density)
#     - Cold experts: 1:4 pruning (25% density) then padded to 2:4 format

#     Args:
#         weight: Dense weight tensor
#         sparsity_type: 'hot' (2:4) or 'cold' (1:4)
#     Returns:
#         (values, indices, metadata): Samoyeds format ready for execution
#     """
#     nrows, ncols = weight.shape
#     # weight_pruned = weight.clone()
#     # Convert to FP32 for CPU operations
#     weight_pruned = weight.float().clone()
    
#     if sparsity_type == 'cold':
#         # 1:4 pruning: keep top-1 per group of 4
#         reshaped = weight_pruned.view(nrows, ncols // 4, 4)
#         abs_vals = reshaped.abs()
#         top1_idx = abs_vals.argmax(dim=2, keepdim=True)
#         mask = torch.zeros_like(reshaped, dtype=torch.bool)
#         mask.scatter_(2, top1_idx, True)
#         reshaped = torch.where(mask, reshaped, torch.zeros_like(reshaped))
#         weight_pruned = reshaped.view(nrows, ncols)
#     else:
#         # HOT: 2:4 pruning - keep top-2 per group of 4
#         reshaped = weight_pruned.view(nrows, ncols // 4, 4)
#         abs_vals = reshaped.abs()
#         _, top2_idx = abs_vals.topk(2, dim=2)
#         mask = torch.zeros_like(reshaped, dtype=torch.bool)
#         mask.scatter_(2, top2_idx, True)
#         reshaped = torch.where(mask, reshaped, torch.zeros_like(reshaped))
#         weight_pruned = reshaped.view(nrows, ncols)
    
#     # Convert back to FP16 for packing
#     weight_pruned = weight_pruned.half()
    
#     # Always pack as 2:4 for hardware (kernel only supports N=2, M=4)
#     # Cold experts have 1:4 pattern (more zeros), but stored in 2:4 format
#     values, indices, metadata = convert_to_samoyed_format(weight_pruned, sparsity_type='hot')
#     return values, indices, metadata

def apply_exact_sparsification(weight: torch.Tensor, 
                               sparsity_type: str = 'hot',
                               use_modelopt: bool = True,
                               calib_data: torch.Tensor = None) -> tuple:
    """
    EXACT sparsification with optional ModelOpt integration
    
    Args:
        weight: Dense weight tensor [out_features, in_features]
        sparsity_type: 'hot' (2:4) or 'cold' (1:4)
        use_modelopt: Use ModelOpt for better pruning (slower but higher quality)
        calib_data: Optional calibration data for SparseGPT
    
    Returns:
        (values, indices, metadata): Samoyeds format
    """
    # nrows, ncols = weight.shape
    
    if sparsity_type == 'hot' and use_modelopt:
        # Use ModelOpt for 2:4 (hardware-accelerated)
        weight_pruned = _prune_with_modelopt_24(weight, calib_data)
    elif sparsity_type == 'cold':
        # Custom 1:4 pruning (no ModelOpt support)
        weight_pruned = _prune_14_magnitude(weight)
    else:
        # Fallback: magnitude-based 2:4
        weight_pruned = _prune_24_magnitude(weight)
    
    # Convert to Samoyeds format
    weight_pruned = weight_pruned.half()
    values, indices, metadata = convert_to_samoyed_format(
        weight_pruned, 
        sparsity_type='hot'  # Always pack as 2:4 for hardware
    )
    return values, indices, metadata


def _prune_with_modelopt_24(weight: torch.Tensor, calib_data: torch.Tensor = None) -> torch.Tensor:
    """Use ModelOpt for 2:4 sparsification (high quality)"""
    import modelopt.torch.sparsity as mts
    
    # Create temporary module wrapper
    class TempLinear(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.linear = nn.Linear(*w.shape[::-1], bias=False)
            self.linear.weight.data = w.clone()
        
        def forward(self, x):
            return self.linear(x)
    
    temp_model = TempLinear(weight).cuda()
    
    # Apply ModelOpt sparsification
    if calib_data is not None:
        # SparseGPT (data-driven, better accuracy)
        config = {
            "data_loader": [(calib_data,)],  # Wrap in dataloader format
            "collect_func": lambda x: x,
        }
        sparse_model = mts.sparsify(temp_model, "sparsegpt", config=config)
    else:
        # Magnitude-based (faster, no calibration needed)
        sparse_model = mts.sparsify(temp_model, "sparse_magnitude")
    
    # Extract pruned weight
    pruned_weight = sparse_model.linear.weight.data.cpu()
    del temp_model, sparse_model
    torch.cuda.empty_cache()
    
    return pruned_weight

def _prune_24_magnitude(weight: torch.Tensor, return_to_device: bool = True) -> torch.Tensor:
    """Magnitude-based 2:4 pruning (fast baseline)
    
    Args:
        weight: Weight tensor to prune
        return_to_device: If False, keep result on CPU (useful for in-place updates)
    """
    device = weight.device
    weight_cpu = weight.cpu().float()

    nrows, ncols = weight_cpu.shape
    weight_float = weight_cpu.clone()
    reshaped = weight_float.view(nrows, ncols // 4, 4)
    abs_vals = reshaped.abs()
    _, top2_idx = abs_vals.topk(2, dim=2)
    mask = torch.zeros_like(reshaped, dtype=torch.bool)
    mask.scatter_(2, top2_idx, True)
    reshaped = torch.where(mask, reshaped, torch.zeros_like(reshaped))
    pruned = reshaped.view(nrows, ncols)
    
    # VERIFY 2:4 pattern
    check = pruned.view(nrows, ncols // 4, 4)
    nnz_per_group = (check != 0).sum(dim=2)
    assert (nnz_per_group == 2).all(), f"Pruning failed: {nnz_per_group.unique()}"
    
    if return_to_device:
        return pruned.to(device)
    return pruned


def _prune_14_magnitude(weight: torch.Tensor) -> torch.Tensor:
    """Magnitude-based 1:4 pruning (EXACT cold experts)"""
    nrows, ncols = weight.shape
    weight_float = weight.float().clone()
    reshaped = weight_float.view(nrows, ncols // 4, 4)
    abs_vals = reshaped.abs()
    top1_idx = abs_vals.argmax(dim=2, keepdim=True)
    mask = torch.zeros_like(reshaped, dtype=torch.bool)
    mask.scatter_(2, top1_idx, True)
    reshaped = torch.where(mask, reshaped, torch.zeros_like(reshaped))
    return reshaped.view(nrows, ncols)

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
