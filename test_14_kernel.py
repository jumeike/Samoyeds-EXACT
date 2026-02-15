"""Test if the CUDA kernel can execute with 1:4 data"""
import torch
import sys
sys.path.insert(0, '/fs/nexus-projects/EXACT/Samoyeds-EXACT/build/lib.linux-x86_64-cpython-310')
import samoyeds_kernel

# Create 1:4 sparse data
N, M = 1, 4
nrows, ncols = 14336, 4096
SPTC_N, SPTC_M = 2, 4
vector_length = 128

# Allocate tensors on GPU
values_rows = nrows // M * N  # 3584
values_cols = ncols // SPTC_M * SPTC_N  # 2048

print(f"Creating 1:4 sparse tensors...")
weight = torch.randn(nrows, ncols, dtype=torch.float16)
A_values = torch.zeros((values_rows, values_cols), dtype=torch.float16)
A_indices = torch.zeros((ncols // vector_length, values_rows), dtype=torch.int32)
A_metadata = torch.zeros((values_rows, values_cols // 16), dtype=torch.int32)

# Call sparsifier
samoyeds_kernel.sparsifier(weight, A_values, A_indices, A_metadata,
                           nrows, ncols, vector_length, N, M)

print(f"Sparsifier output shapes:")
print(f"  values: {A_values.shape}")
print(f"  indices: {A_indices.shape}")  
print(f"  metadata: {A_metadata.shape}")

# Move to GPU
A_values = A_values.cuda()
A_indices = A_indices.cuda()
A_metadata = A_metadata.cuda()

# Create dense input (batch_size=56 from the error log)
batch_size = 56
input_dense = torch.randn(batch_size, ncols, dtype=torch.float16).cuda()

# Create index tensor
idx = torch.arange(batch_size, dtype=torch.int32).cuda()

print(f"\nCalling CUDA kernel with 1:4 data...")
print(f"  A_values: {A_values.shape}, device={A_values.device}")
print(f"  input: {input_dense.shape}, device={input_dense.device}")

try:
    # This should match SSTransLinear forward dimensions
    output_rows = values_rows * (M // N)  # 3584 * 4 = 14336
    output_cols = values_cols * (M // N)  # 2048 * 4 = 8192
    
    # Call the kernel directly (bypass the dispatch wrapper)
    print(f"Calling samoyeds_kernel.ssmm_trans directly...")
    output = samoyeds_kernel.ssmm_trans(
        A_values, A_indices, A_metadata,  # sparse weights
        input_dense, idx,  # dense input 
        output_rows, output_cols, batch_size,  # dimensions
        vector_length,  # vector_length=128
        N, M  # structure params: 1, 4
    )
    print(f"✓ Kernel succeeded! Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Kernel failed: {e}")
    import traceback
    traceback.print_exc()
