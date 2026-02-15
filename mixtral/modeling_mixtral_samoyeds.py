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

import torch
import torch.nn as nn
import torch.nn.functional as F

from mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralBLockSparseTop2MLP

from module.linear.SSFusedSiluTransLinear import SSFusedSiluTransLinear
from module.linear.SPDenseWeightedTransLinear import SPDenseWeightedLinear
from module.linear.SSTransLinear import SSTransLinear

class SSMixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, original: MixtralBLockSparseTop2MLP, skip_sparsifier=False):
        super().__init__()
        self.ffn_dim = original.ffn_dim
        self.hidden_dim = original.hidden_dim

        if skip_sparsifier:
            # Create dummy tensors - EXACT will replace them
            self.w1 = SSFusedSiluTransLinear(None, in_features=self.hidden_dim, out_features=self.ffn_dim, skip_sparsifier=True)
            self.w2 = SPDenseWeightedLinear(None, in_features=self.ffn_dim, out_features=self.hidden_dim, skip_sparsifier=True)
            self.w3 = SSTransLinear(None, in_features=self.hidden_dim, out_features=self.ffn_dim, skip_sparsifier=True)
        else:
            # w1 w3是SS的矩阵乘，input带有expert选择，hidden_states是完整的中间结果，返回结果为压缩的稠密矩阵
            # w2是SD的矩阵乘
            # w1 w2 w3 的权重都会根据要求进行稀疏化裁剪
            self.w1 = SSFusedSiluTransLinear(nn.Linear(self.hidden_dim, self.ffn_dim, bias=False))
            self.w2 = SPDenseWeightedLinear(nn.Linear(self.ffn_dim, self.hidden_dim, bias=False))
            self.w3 = SSTransLinear(nn.Linear(self.hidden_dim, self.ffn_dim, bias=False))

        self.act_fn = original.act_fn

    def forward(self, hidden_states, input_idx, routing_weights):
        # print(f"[DEBUG] hidden_states.shape={hidden_states.shape}, input_idx.shape={input_idx.shape}, input_idx.dtype={input_idx.dtype}")
        # print(f"[DEBUG] input_idx min={input_idx.min()}, max={input_idx.max()}")        
        batch_size = input_idx.shape[0]
        current_hidden_states = self.w1(hidden_states, input_idx) * self.w3(hidden_states, input_idx)
        current_hidden_states = self.w2(current_hidden_states, routing_weights.T)
        return current_hidden_states[0:batch_size, :]


class SSMixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, original: MixtralSparseMoeBlock, skip_sparsifier=False):
        super().__init__()
        self.hidden_dim = original.hidden_dim
        self.ffn_dim = original.ffn_dim
        self.num_experts = original.num_experts
        self.top_k = original.top_k

        # gating
        self.gate = original.gate

        self.experts = nn.ModuleList([SSMixtralBLockSparseTop2MLP(expert, skip_sparsifier=skip_sparsifier) for expert in original.experts])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # Force synchronization
            torch.cuda.synchronize()

            top_x = top_x.clone()
            idx = idx.clone()

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            torch.cuda.synchronize()
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            # current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            # current_hidden_states = expert_layer(current_state, routing_weights[top_x_list, idx_list, None])

            # 实际计算时hidden_states的选择在算子中进行
            current_hidden_states = expert_layer(hidden_states, top_x, routing_weights[top_x_list, idx_list, None])
            # expert_layer(hidden_states, top_x, routing_weights[top_x_list, idx_list, None])
            torch.cuda.synchronize()
            # Adaptive padding: cold experts (1:4) output smaller dims, pad to match hot (2:4)
            if current_hidden_states.shape[-1] != hidden_dim:
                # Cold expert output needs padding
                pad_size = hidden_dim - current_hidden_states.shape[-1]
                current_hidden_states = torch.nn.functional.pad(current_hidden_states, (0, pad_size), value=0.0)
            
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            # final_hidden_states = final_hidden_states + current_hidden_states.to(hidden_states.dtype)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

def sparsemoeblock_to_ss(mod, skip_sparsifier=False):
    # Check by class name to handle different import paths (transformers vs local)
    if type(mod).__name__ == 'MixtralSparseMoeBlock':
        # print("in module replace")
        return SSMixtralSparseMoeBlock(mod, skip_sparsifier=skip_sparsifier)
    for name, m in mod.named_children():
        if isinstance(m, SSMixtralSparseMoeBlock):
            continue
        # if isinstance(m, torch.nn.Linear):
        if type(m).__name__ == 'MixtralSparseMoeBlock':
            setattr(mod, name, SSMixtralSparseMoeBlock(m, skip_sparsifier=skip_sparsifier))
        elif m is not mod:
            sparsemoeblock_to_ss(m, skip_sparsifier=skip_sparsifier)

    return mod