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


def _expand_cold_metadata_1_4_to_2_4(compact_meta: torch.Tensor) -> torch.Tensor:
    """
    Expand compact 1:4 metadata (16x2-bit entries per uint) into hardware-ready
    2:4 metadata words (8x4-bit entries per uint).

    Output has 2x metadata columns vs compact input.
    """
    # pair_bits for [val, 0] payload ordering:
    # pos0 -> (0,1)=0x4, pos1 -> (1,0)=0x1, pos2 -> (2,3)=0xE, pos3 -> (3,2)=0xB
    lut = torch.tensor([0x4, 0x1, 0xE, 0xB], dtype=torch.int64)

    meta64 = compact_meta.to(torch.int64) & 0xFFFFFFFF
    low_half = meta64 & 0xFFFF
    high_half = (meta64 >> 16) & 0xFFFF

    def expand_half(h16: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(h16, dtype=torch.int64)
        for i in range(8):
            pos = (h16 >> (2 * i)) & 0x3
            out |= lut[pos] << (4 * i)
        return out

    out0 = expand_half(low_half)
    out1 = expand_half(high_half)

    expanded = torch.empty((compact_meta.size(0), compact_meta.size(1) * 2), dtype=torch.int32)
    expanded[:, 0::2] = out0.to(torch.int32)
    expanded[:, 1::2] = out1.to(torch.int32)
    return expanded


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
    # Hot metadata is already hardware-ready. Cold metadata from sparsifier is compact 1:4,
    # so allocate compact first and expand offline to hardware-ready shape.
    if sparsity_type == 'cold':
        compact_meta_cols = values_cols // NUM_OF_META_PER_UINT  # K/4 domain
        A_metadata_compact = torch.zeros((values_rows, compact_meta_cols),
                                         dtype=torch.int32, device='cpu')
        A_metadata = None
    else:
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
    if sparsity_type == 'cold':
        samoyeds_kernel.sparsifier(
            weight_cpu,
            A_values,
            A_indices,
            A_metadata_compact,
            nrows, ncols, vector_length, N, M
        )
        A_metadata = _expand_cold_metadata_1_4_to_2_4(A_metadata_compact)
    else:
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
    # print(f"[DEBUG] values.shape={A_values.shape}, expected=({nrows}, {ncols//M*N})")
    # print(f"[DEBUG] indices.shape={A_indices.shape}, metadata.shape={A_metadata.shape}")
    
    return A_values, A_indices, A_metadata

def apply_exact_sparsification(weight: torch.Tensor, 
                               sparsity_type: str = 'hot',
                               use_modelopt: bool = False,
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

    
    # if sparsity_type == 'hot' and use_modelopt:
    #     # Use ModelOpt for 2:4 (hardware-accelerated)
    #     weight_pruned = _prune_with_modelopt_24(weight, calib_data)
    # elif sparsity_type == 'cold':
    #     # COLD: 1:4 pruning 
    #     weight_pruned = _prune_14_magnitude(weight)
    # else:
    #     # HOT: 2:4 pruning - keep top-2 per group of 4
    #     weight_pruned = _prune_24_magnitude(weight)
    
    # # Convert back to FP16 for packing
    # weight_pruned = weight_pruned.half()
    
    # Pack as 2:4 format for hardware
    values, indices, metadata = convert_to_samoyed_format(
        # weight_pruned, 
        weight,
        sparsity_type=sparsity_type
    )
    return values, indices, metadata


# def _prune_with_modelopt_24(weight: torch.Tensor, calib_data: torch.Tensor = None) -> torch.Tensor:
#     """Use ModelOpt for 2:4 sparsification (high quality)"""
#     import modelopt.torch.sparsity as mts
    
#     # Create temporary module wrapper
#     class TempLinear(nn.Module):
#         def __init__(self, w):
#             super().__init__()
#             self.linear = nn.Linear(*w.shape[::-1], bias=False)
#             self.linear.weight.data = w.clone()
        
#         def forward(self, x):
#             return self.linear(x)
    
#     temp_model = TempLinear(weight).cuda()
    
#     # Apply ModelOpt sparsification
#     if calib_data is not None:
#         # SparseGPT (data-driven, better accuracy)
#         config = {
#             "data_loader": [(calib_data,)],  # Wrap in dataloader format
#             "collect_func": lambda x: x,
#         }
#         sparse_model = mts.sparsify(temp_model, "sparsegpt", config=config)
#     else:
#         # Magnitude-based (faster, no calibration needed)
#         sparse_model = mts.sparsify(temp_model, "sparse_magnitude")
    
#     # Extract pruned weight
#     pruned_weight = sparse_model.linear.weight.data.cpu()
#     del temp_model, sparse_model
#     torch.cuda.empty_cache()
    
#     return pruned_weight

def _prune_with_modelopt_24(weight: torch.Tensor, calib_data=None) -> torch.Tensor:
    import modelopt.torch.sparsity as mts
    import torch.nn as nn

    class TempLinear(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.linear = nn.Linear(w.shape[1], w.shape[0], bias=False)
            self.linear.weight.data.copy_(w)

        def forward(self, x):
            return self.linear(x)

    device = torch.device("cuda")
    temp_model = TempLinear(weight.detach().to(device)).eval()

    with torch.no_grad():
        if calib_data is not None:
            sparse_model = mts.sparsify(
                temp_model,
                "sparsegpt",
                config={"data_loader": [(calib_data.to(device),)]}
            )
        else:
            sparse_model = mts.sparsify(temp_model, "sparse_magnitude")

    pruned = sparse_model.linear.weight.detach().cpu()
    del temp_model, sparse_model
    torch.cuda.empty_cache()
    return pruned

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


def _prune_14_magnitude(weight: torch.Tensor, return_to_device: bool = True) -> torch.Tensor:
    """Magnitude-based 1:4 pruning (EXACT cold experts)

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
    top1_idx = abs_vals.argmax(dim=2, keepdim=True)
    mask = torch.zeros_like(reshaped, dtype=torch.bool)
    mask.scatter_(2, top1_idx, True)
    reshaped = torch.where(mask, reshaped, torch.zeros_like(reshaped))
    pruned = reshaped.view(nrows, ncols)

    # Optional verify 1:4 pattern
    # check = pruned.view(nrows, ncols // 4, 4)
    # nnz_per_group = (check != 0).sum(dim=2)
    # assert (nnz_per_group == 1).all(), f"1:4 pruning failed: {nnz_per_group.unique()}"

    if return_to_device:
        return pruned.to(device)
    return pruned

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
