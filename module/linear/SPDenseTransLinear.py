#
# Copyright (c) 2025 Chenpeng Wu (cpwu_sjtu@sjtu.edu.cn), Qiqi Gu (qiqi.gu@sjtu.edu.cn). 
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch.nn as nn
import torch

from module.util import sparsifier, N, M, vector_length
import samoyeds_kernel

def sparse_dense_mul_transpose_dense_dispatch(A_sparse_values, A_sparse_indices, A_sparse_metadata,
                                                B_values, m, k, n, structure_N=None, structure_M=None):
    # Use provided N, M or fall back to global defaults
    if structure_N is None:
        structure_N = N
    if structure_M is None:
        structure_M = M
    # print(f"[sparse_dense_mul_transpose_dense_dispatch] N={structure_N}, M={structure_M}")
    output = samoyeds_kernel.spmm_dense_trans( A_sparse_values,  # lhs_matrix, compressed values
                                   A_sparse_indices,  # indices
                                   A_sparse_metadata,  # metadata
                                   B_values,  # rhs_matrix, sparse inputs
                                   m,  # m
                                   k,  # k
                                   n,  # n
                                   vector_length,  # vector_length
                                   structure_N,  # structure_N
                                   structure_M,  # structure_M
                                   )

    return output

class SPDenseTransLinear(nn.Module):
    def __init__(self, original = None, in_features = 1024, out_features = 4096, skip_sparsifier=False):
        super().__init__()
        if original is None:
            self.in_features = in_features
            self.out_features = out_features
            if skip_sparsifier:
                # Create dummy tensors - EXACT will replace them
                self.weight = nn.Parameter(torch.empty(out_features//2, in_features//2))
                self.indices = nn.Parameter(torch.empty(in_features//128, out_features//2), requires_grad=False)
                self.metadata = nn.Parameter(torch.empty(out_features//2, in_features//32), requires_grad=False)
            else:
                self.weight, self.indices, self.metadata = sparsifier(nn.Parameter(torch.rand(out_features, in_features)))
        else:
            self.bias = original.bias
            self.in_features = original.in_features
            self.out_features = original.out_features
            self.weight, self.indices, self.metadata = sparsifier(original.weight)

        self.weight = nn.Parameter(self.weight)
        self.indices = nn.Parameter(self.indices, requires_grad=False)
        self.metadata = nn.Parameter(self.metadata, requires_grad=False)
        
        # N:M sparsity parameters (default to Samoyeds baseline 1:2)
        # EXACT will update these for variable compression
        self.N = N
        self.M = M

    def forward(self, input):
        # print(f"[EXACT] SPDenseTransLinear forward path: weight={self.weight.shape}, N={self.N}, M={self.M}")
        return sparse_dense_mul_transpose_dense_dispatch(self.weight, self.indices, self.metadata, input,
                                          self.weight.shape[0] * 2,
                                          self.weight.shape[1] * 2, input.shape[0],
                                          self.N, self.M)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )