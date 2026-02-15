# Samoyed Kernel Call Trace for DeepSeek MoE

This document traces the execution path from model evaluation down to Samoyed sparse kernels.

## Current Baseline Uses DENSE MoE (Not Samoyed)

**IMPORTANT**: The baseline evaluation you ran uses `DeepseekMoE` (dense), NOT `SSDeepseekMoE` (Samoyed sparse).

## Two Implementations in Codebase

### 1. Dense MoE (HuggingFace Standard)
**File**: `deepseek/modeling_deepseek.py`
```python
class DeepseekMoE(nn.Module):
    def __init__(self, config):
        self.experts = nn.ModuleList([
            DeepseekMLP(config, ...) for i in range(config.n_routed_experts)
        ])
    
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        for i, end_idx in enumerate(tokens_per_expert):
            expert = self.experts[i]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)  # ← Standard PyTorch nn.Linear
            expert_cache.scatter_reduce_(...)
```

**Expert Implementation**: `DeepseekMLP`
- Uses standard `nn.Linear` layers
- gate_proj, up_proj, down_proj are dense matrices
- No custom kernels

---

### 2. Samoyed Sparse MoE 
**File**: `deepseek/modeling_deepseek_samoyeds.py`
```python
class SSDeepseekMoE(nn.Module):
    def __init__(self, config):
        self.experts = nn.ModuleList([
            SSDeepseekMLP(config, ...) for i in range(config.n_routed_experts)
        ])
    
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        for i, end_idx in enumerate(tokens_per_expert):
            expert = self.experts[i]
            expert_out = expert(x, exp_token_idx, flat_expert_weights[...])
            # ↑ Passes indices to Samoyed sparse expert
            expert_cache.scatter_reduce_(...)
```

**Expert Implementation**: `SSDeepseekMLP`
```python
class SSDeepseekMLP(nn.Module):
    def __init__(self, config, ...):
        # SAMOYED SPARSE LAYERS
        self.gate_proj = SSFusedSiluTransLinear(...)  # ← Sparse!
        self.up_proj = SSTransLinear(...)             # ← Sparse!
        self.down_proj = SPDenseWeightedLinear(...)   # ← Sparse!
    
    def forward(self, hidden_states, input_idx, routing_weights):
        # Uses sparse matrix operations
        current_hidden_states = self.gate_proj(hidden_states, input_idx) * \
                                self.up_proj(hidden_states, input_idx)
        return self.down_proj(current_hidden_states, routing_weights.T)
```

---

## Complete Call Stack for Samoyed (When Used)

### Level 1: Model Forward Pass
```
evaluate_samoyed_baseline_deepseek.py
  └─ lm_eval evaluator.simple_evaluate()
      └─ HFLM._model_call(inputs)
          └─ model(inputs).logits
```

### Level 2: Model Architecture
**File**: `deepseek/modeling_deepseek.py:1330`
```python
class DeepseekForCausalLM:
    def forward(self, input_ids, ...):
        outputs = self.model(input_ids, ...)  # DeepseekModel
        logits = self.lm_head(outputs[0])
```

### Level 3: Layer Iteration
**File**: `deepseek/modeling_deepseek.py:1217`
```python
class DeepseekModel:
    def forward(self, input_ids, ...):
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(hidden_states, ...)
```

### Level 4: Single Layer (MoE or Dense)
**File**: `deepseek/modeling_deepseek.py:959` OR `deepseek/modeling_deepseek_samoyeds.py`
```python
class DeepseekDecoderLayer:  # or SSDeepseekDecoderLayer
    def forward(self, hidden_states, ...):
        # Attention
        hidden_states = self.self_attn(hidden_states, ...)
        
        # MoE or FFN
        hidden_states = self.mlp(hidden_states)  # ← MoE block
```

### Level 5: MoE Block
**File**: `deepseek/modeling_deepseek_samoyeds.py:155` (Samoyed version)
```python
class SSDeepseekMoE:
    def forward(self, hidden_states):
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        
        if self.training:
            # Training path (not relevant for eval)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight)
```

### Level 6: MoE Inference Loop
**File**: `deepseek/modeling_deepseek_samoyeds.py:177`
```python
class SSDeepseekMoE:
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        for i, end_idx in enumerate(tokens_per_expert):
            expert = self.experts[i]  # SSDeepseekMLP
            exp_token_idx = token_idxs[start_idx:end_idx]
            
            # KEY: Passes indices to sparse expert!
            expert_out = expert(x, exp_token_idx, flat_expert_weights[...])
            expert_cache.scatter_reduce_(...)
```

### Level 7: Sparse Expert Forward
**File**: `deepseek/modeling_deepseek_samoyeds.py:93`
```python
class SSDeepseekMLP:
    def forward(self, hidden_states, input_idx, routing_weights):
        # Samoyed sparse operations
        current_hidden_states = self.gate_proj(hidden_states, input_idx) * \
                                self.up_proj(hidden_states, input_idx)
        return self.down_proj(current_hidden_states, routing_weights.T)
```

### Level 8: Samoyed Sparse Linear Layer
**File**: `module/linear/SSFusedSiluTransLinear.py:71`
```python
class SSFusedSiluTransLinear:
    def forward(self, input, idx):
        new_idx = padding_idx(idx, 64)
        output = sparse_sparse_mul_fused_silu_trans_dispatch(
            self.weight,    # Sparse values (N:M format)
            self.indices,   # Sparse indices
            self.metadata,  # Sparse metadata
            input,          # Dense input
            new_idx,        # Token indices
            self.N,         # Sparsity N (e.g., 2 for 2:4)
            self.M          # Sparsity M (e.g., 4 for 2:4)
        )
```

### Level 9: C++ Kernel Dispatcher
**File**: `module/linear/SSFusedSiluTransLinear.py:23`
```python
def sparse_sparse_mul_fused_silu_trans_dispatch(...):
    output = samoyeds_kernel.ssmm_fused_silu_trans(
        A_sparse_values,   # Compressed weight values
        A_sparse_indices,  # Which elements are non-zero
        A_sparse_metadata, # Row/column metadata
        B_values,          # Input activations
        B_indices,         # Token selection indices
        m, k, n,           # Matrix dimensions
        vector_length,     # SIMD vector length
        structure_N,       # 2 (for 2:4 sparsity)
        structure_M        # 4 (for 2:4 sparsity)
    )
```

### Level 10: CUDA Kernel
**File**: `Samoyeds-Kernel/csrc/ssmm_fused_silu.cu` (compiled C++/CUDA)
```cpp
// Sparse-Sparse Matrix Multiplication with Fused SiLU
// This is the actual GPU kernel that performs:
// 1. Sparse weight access (N:M structured sparsity)
// 2. Sparse input selection (token indices)
// 3. Matrix multiplication
// 4. Fused SiLU activation
// All in a single optimized CUDA kernel
```

---

## How to Verify Which Implementation is Running

### Method 1: Check Model Class Type
```python
# In your evaluation script, add:
print(f"Model type: {type(samoyeds_model)}")
print(f"First layer MLP type: {type(samoyeds_model.model.layers[0].mlp)}")

# Expected outputs:
# Samoyed: <class 'deepseek.modeling_deepseek_samoyeds.SSDeepseekMoE'>
# Dense:   <class 'deepseek.modeling_deepseek.DeepseekMoE'>
```

### Method 2: Check Expert Type
```python
# Check the expert implementation
first_moe_layer = samoyeds_model.model.layers[0].mlp
if hasattr(first_moe_layer, 'experts'):
    expert = first_moe_layer.experts[0]
    print(f"Expert type: {type(expert)}")
    print(f"gate_proj type: {type(expert.gate_proj)}")

# Expected outputs:
# Samoyed expert: <class 'deepseek.modeling_deepseek_samoyeds.SSDeepseekMLP'>
# Samoyed gate:   <class 'module.linear.SSFusedSiluTransLinear.SSFusedSiluTransLinear'>
# Dense expert:   <class 'deepseek.modeling_deepseek.DeepseekMLP'>
# Dense gate:     <class 'torch.nn.modules.linear.Linear'>
```

### Method 3: Check for Sparse Attributes
```python
# Samoyed layers have special sparse attributes
layer = samoyeds_model.model.layers[0].mlp.experts[0].gate_proj
if hasattr(layer, 'indices') and hasattr(layer, 'metadata'):
    print("✓ This is Samoyed sparse format!")
    print(f"  N:M ratio: {layer.N}:{layer.M}")
else:
    print("✗ This is dense format (standard PyTorch)")
```

---

## Current Issue in evaluate_samoyed_baseline_deepseek.py

**Problem**: The script creates `DeepseekForCausalLM` which uses `DeepseekMoE` (dense) by default.

**Solution**: To actually use Samoyed, you need to either:

1. **Option A**: Use the exact same flow as `evaluate_exact_deepseek.py`:
   - Load dense model
   - Convert to `SSDeepseekMoE` manually (what you tried but got CUDA error)
   - Apply EXACT compression (which makes it truly sparse)

2. **Option B**: Create a model that uses `SSDeepseekDecoderLayer` from the start:
   ```python
   # Would require custom model initialization
   # This is NOT straightforward with HF pretrained weights
   ```

3. **Option C**: Compare dense HF baseline vs EXACT-compressed Samoyed:
   - Dense: Standard HF transformers (what you have now)
   - Sparse: evaluate_exact_deepseek.py output
   - This is the most common comparison

---

## Why Your Current Script Gets 25.66% Accuracy

The 25.66% MMLU accuracy is reasonable for DeepSeek-MoE-16B using the **dense implementation**. 

This is NOT proof of Samoyed working - it's just the standard HuggingFace implementation loaded with your custom `DeepseekForCausalLM` class (which defaults to dense `DeepseekMoE`).

To actually test Samoyed baseline, you'd need to use the sparse kernel path, but that requires:
1. Converting to `SSDeepseekMoE`
2. Ensuring weights are in sparse format (N:M structured)
3. Or just use EXACT which does all of this automatically

---

## Recommendation

**Don't try to evaluate "Samoyed baseline" separately.** The proper comparison is:

1. **Dense Baseline**: HuggingFace transformers (your current script ✓)
2. **EXACT Compressed**: `evaluate_exact_deepseek.py` which uses Samoyed kernels with sparse weights

The "Samoyed baseline" concept doesn't make sense because:
- Samoyed kernels expect sparse weights (N:M format)
- Pretrained weights are dense
- EXACT is what converts dense → sparse in a smart way
- Without EXACT, you'd just be using regular dense operations

Your current script is fine - it's comparing dense HF vs EXACT-compressed models, which is the right comparison.
