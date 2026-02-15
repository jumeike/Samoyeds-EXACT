#!/usr/bin/env python3
import argparse
import itertools

import torch
import torch.nn.functional as F

from module.linear.SSFusedSiluTransLinear import SSFusedSiluTransLinear
from module.util import NUM_OF_META_PER_UINT, vector_length, padding_idx


def fill_metadata_all_pos(meta_tensor: torch.Tensor, pos: int) -> None:
    # 2-bit packed metadata, 16 entries per uint32.
    # Use signed int32 literals so torch.int32 fill_ does not overflow.
    lut = [0, 0x55555555, -1431655766, -1]  # 0xAAAAAAAA, 0xFFFFFFFF as int32
    meta_tensor.fill_(lut[pos])


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def reconstruct_dense_weight_14(
    values: torch.Tensor, indices: torch.Tensor, metadata: torch.Tensor, m: int, k: int
) -> torch.Tensor:
    # Rebuild dense A (m x k) from 1:4 storage used by this check:
    # values: [m/4, k/4], indices: [k/vector_length, m/4], metadata: [m/4, (k/4)/16]
    dense = torch.zeros((m, k), device=values.device, dtype=values.dtype)
    value_rows, value_cols = values.shape
    groups_per_vector = vector_length // 4

    for r in range(value_rows):
        for c in range(value_cols):
            idx_col = c // groups_per_vector
            selected_index = int(indices[idx_col, r].item())
            row_offset = r * 4 + selected_index

            meta_u32 = int(metadata[r, c // NUM_OF_META_PER_UINT].item()) & 0xFFFFFFFF
            pos = (meta_u32 >> ((c % NUM_OF_META_PER_UINT) * 2)) & 0x3

            dense[row_offset, c * 4 + pos] = values[r, c]
    return dense


def align_and_max_abs_diff(out: torch.Tensor, ref: torch.Tensor) -> float:
    if out.shape == ref.shape:
        return max_abs_diff(out, ref)
    if out.t().shape == ref.shape:
        return max_abs_diff(out.t(), ref)
    raise RuntimeError(f"shape mismatch: out={tuple(out.shape)}, ref={tuple(ref.shape)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=128, help="Input feature size (must be multiple of 128).")
    parser.add_argument("--m", type=int, default=512, help="Output feature size for kernel launch.")
    parser.add_argument("--tokens", type=int, default=64, help="Input rows (token count).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda.")
    args = parser.parse_args()

    if args.k % vector_length != 0:
        raise ValueError(f"--k must be multiple of vector_length={vector_length}")
    if args.m % 4 != 0:
        raise ValueError("--m must be multiple of 4 for logical 1:4 test")
    # For SSMM launch in 1:4 path: dim_x = m / 128 / 4 * 1, so m must be >= 512.
    # To avoid silent truncation on partial blocks, require 512 alignment.
    if args.m < 512 or args.m % 512 != 0:
        raise ValueError("--m must be >= 512 and divisible by 512 for this 1:4 kernel check")
    if args.tokens <= 0:
        raise ValueError("--tokens must be > 0")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16

    # Build layer without default 1:2 sparsifier path.
    layer = SSFusedSiluTransLinear(in_features=args.k, out_features=args.m, skip_sparsifier=True).to(device)
    layer.N = 1
    layer.M = 4

    # Allocate 1:4 storage shapes explicitly for this test.
    ratio = layer.M // layer.N  # 4
    value_rows = args.m // ratio
    value_cols = args.k // ratio
    meta_cols = (args.k // layer.M * layer.N) // NUM_OF_META_PER_UINT
    index_rows = args.k // vector_length
    index_cols = value_rows

    layer.weight = torch.nn.Parameter(torch.randn((value_rows, value_cols), device=device, dtype=dtype) * 0.1)
    layer.indices = torch.nn.Parameter(
        torch.zeros((index_rows, index_cols), device=device, dtype=torch.int32), requires_grad=False
    )
    layer.metadata = torch.nn.Parameter(
        torch.zeros((value_rows, meta_cols), device=device, dtype=torch.int32), requires_grad=False
    )

    x = torch.randn((args.tokens, args.k), device=device, dtype=dtype) * 0.1
    idx = torch.arange(args.tokens, device=device, dtype=torch.int32)

    # 1) Metadata sensitivity: random metadata A vs B with same values/indices/input.
    with torch.no_grad():
        layer.metadata.copy_(torch.randint(0, 2**31 - 1, layer.metadata.shape, device=device, dtype=torch.int32))
        out_a = layer(x, idx)
        layer.metadata.copy_(torch.randint(0, 2**31 - 1, layer.metadata.shape, device=device, dtype=torch.int32))
        out_b = layer(x, idx)
    sensitivity = max_abs_diff(out_a, out_b)
    print(f"[check] metadata sensitivity max|diff| = {sensitivity:.6f}")

    # 2) Forced pos sweep (all metadata entries set to the same pos in each run).
    outs = {}
    with torch.no_grad():
        for pos in [0, 1, 2, 3]:
            fill_metadata_all_pos(layer.metadata, pos)
            outs[pos] = layer(x, idx).clone()

    print("[check] forced pos pairwise max|diff|")
    pair_diffs = []
    for i, j in itertools.combinations([0, 1, 2, 3], 2):
        d = max_abs_diff(outs[i], outs[j])
        pair_diffs.append(d)
        print(f"  pos{i} vs pos{j}: {d:.6f}")

    changed = sensitivity > 0.0 and max(pair_diffs) > 0.0
    if not changed:
        print("[result] FAIL")
        print("[hint] Outputs did not react to metadata changes; metadata path may still be ineffective.")
        return 1

    # 3) Dense-reference check for forced positions.
    # Kernel path here is fused silu trans, so reference is silu(B_selected @ A_dense^T).
    print("[check] dense reference max|err| per forced pos")
    ref_errs = []
    with torch.no_grad():
        new_idx = padding_idx(idx, 64).to(torch.long)
        b_selected = x[new_idx]
        for pos in [0, 1, 2, 3]:
            fill_metadata_all_pos(layer.metadata, pos)
            out = layer(x, idx)
            a_dense = reconstruct_dense_weight_14(layer.weight, layer.indices, layer.metadata, args.m, args.k)
            ref = F.silu(torch.matmul(b_selected, a_dense.t()))
            err = align_and_max_abs_diff(out, ref)
            ref_errs.append(err)
            print(f"  pos{pos}: {err:.6f}")

    ref_ok = max(ref_errs) < 1e-2
    print(f"[result] {'PASS' if ref_ok else 'FAIL'} (sensitivity + dense check)")
    if not ref_ok:
        print("[hint] Metadata is active, but numerical match to dense reference is off.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
