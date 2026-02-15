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

from module.util import sparsifier, padding_idx, N, M, vector_length, NUM_OF_META_PER_UINT
import samoyeds_kernel

def sparse_sparse_mul_trans_dispatch(A_sparse_values, A_sparse_indices, A_sparse_metadata, B_values, B_indices, m, k, n, structure_N=None, structure_M=None):
    # Use provided N, M or fall back to global defaults
    if structure_N is None:
        structure_N = N
    if structure_M is None:
        structure_M = M
    # print(f"[sparse_sparse_mul_trans_dispatch] N={structure_N}, M={structure_M}")    
    output = samoyeds_kernel.ssmm_trans(   A_sparse_values,  # lhs_matrix, compressed values
                                    A_sparse_indices,  # indices
                                    A_sparse_metadata,  # metadata
                                    B_values,  # rhs_matrix, sparse inputs
                                    B_indices,  # indices, indices selection
                                    m,  # m
                                    k,  # k
                                    n,  # n
                                    vector_length,  # vector_length
                                    structure_N,  # structure_N
                                    structure_M  # structure_M
                                )

    return output

class SSTransLinear(nn.Module):
    def __init__(self, original = None, in_features = 1024, out_features = 4096, skip_sparsifier=False):
        super().__init__()
        if original is None:
            self.in_features = in_features
            self.out_features = out_features
            self.original_out_features = out_features  # Store original dense dim
            self.original_in_features = in_features
            # N:M sparsity parameters - set before creating tensors if adaptive
            self.N = N
            self.M = M
            if skip_sparsifier:
                # Create dummy tensors using N:M ratio
                ratio = self.M // self.N
                self.weight = nn.Parameter(torch.empty(out_features//ratio, in_features//ratio))
                self.indices = nn.Parameter(torch.empty(in_features//vector_length, out_features//ratio), requires_grad=False)
                self.metadata = nn.Parameter(torch.empty(out_features//ratio, in_features//self.M*self.N//NUM_OF_META_PER_UINT), requires_grad=False)
            else:
                self.weight, self.indices, self.metadata = sparsifier(nn.Parameter(torch.rand(out_features, in_features)))
        else:
            self.bias = original.bias
            self.in_features = original.in_features
            self.out_features = original.out_features
            self.original_out_features = original.out_features  # Store dense dimensions
            self.original_in_features = original.in_features
            self.weight, self.indices, self.metadata = sparsifier(original.weight)
            
        self.weight = nn.Parameter(self.weight)
        self.indices = nn.Parameter(self.indices, requires_grad=False)
        self.metadata = nn.Parameter(self.metadata, requires_grad=False)
        
        # N:M sparsity parameters (set earlier if skip_sparsifier, otherwise here)
        if not hasattr(self, 'N'):
            self.N = N
            self.M = M

    def forward(self, input, idx):
        # print('SSTransLinear forward: weight:', self.weight.shape, ', input:', input.shape, ', idx:', idx.shape)
        # new_idx = torch.arange(int(input.shape[0]), device=input.device)
        # idx = torch.arange(int(input.shape[0]), device=input.device)
        # if self.N == 1:  # Cold expert
        # print(f"[EXACT] SSTransLinear path: weight={self.weight.shape}, N={self.N}, M={self.M}")
        new_idx = padding_idx(idx, 64)
        output = sparse_sparse_mul_trans_dispatch(self.weight, self.indices, self.metadata, input, new_idx,
                                                   self.original_out_features,
                                                   self.original_in_features, input.shape[0],
                                                   self.N, self.M)
        # print('SSTransLinear output: ', output[:, 0:idx.shape[0]].shape)
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
