"""Test if samoyeds_kernel.sparsifier supports 1:4 sparsity"""
import torch
import sys
sys.path.insert(0, '/fs/nexus-projects/EXACT/Samoyeds-EXACT/build/lib.linux-x86_64-cpython-310')
import samoyeds_kernel

# Test parameters
N, M = 1, 4
SPTC_N, SPTC_M = 2, 4
vector_length = 128
nrows, ncols = 14336, 4096

# Create test tensor
weight = torch.randn(nrows, ncols, dtype=torch.float16)

# Allocate output tensors (same formula as converter)
values_rows = nrows // M * N  # 14336 // 4 * 1 = 3584
values_cols = ncols // SPTC_M * SPTC_N  # 4096 // 4 * 2 = 2048

print(f"Input: {weight.shape}, N={N}, M={M}")
print(f"Output dims: values=({values_rows}, {values_cols})")

A_values = torch.zeros((values_rows, values_cols), dtype=torch.float16)
A_indices = torch.zeros((ncols // vector_length, values_rows), dtype=torch.int32)
A_metadata = torch.zeros((values_rows, values_cols // 16), dtype=torch.int32)

print(f"Calling sparsifier...")
try:
    samoyeds_kernel.sparsifier(
        weight, A_values, A_indices, A_metadata,
        nrows, ncols, vector_length, N, M
    )
    print(f"✓ Sparsifier succeeded!")
    print(f"Output NaN count: {torch.isnan(A_values).sum().item()}")
except Exception as e:
    print(f"✗ Sparsifier failed: {e}")
