#!/usr/bin/env python3
"""Quick sanity check: test if functions are deterministic"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exact_core import EXACTSparsifier, convert_to_samoyed_format, apply_exact_sparsification

weight = torch.randn(14336, 4096)

print("Test 1: Stage-by-stage twice")
sparsifier = EXACTSparsifier()
sparse1 = sparsifier.apply_2_4_sparsity(weight)
v1, i1, m1 = convert_to_samoyed_format(sparse1, 'hot')

sparse2 = sparsifier.apply_2_4_sparsity(weight)
v2, i2, m2 = convert_to_samoyed_format(sparse2, 'hot')

print(f"  Sparse equal: {torch.equal(sparse1, sparse2)}")
print(f"  Values equal: {torch.equal(v1, v2)}")

print("\nTest 2: Combined pipeline twice")
v3, i3, m3 = apply_exact_sparsification(weight, 'hot')
v4, i4, m4 = apply_exact_sparsification(weight, 'hot')

print(f"  Values equal: {torch.equal(v3, v4)}")
print(f"  Combined vs stage-by-stage: {torch.equal(v1, v3)}")
