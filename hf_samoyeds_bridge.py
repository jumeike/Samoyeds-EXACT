"""
Bridge between HF's calling convention and Samoyeds sparse layers.
Wraps Samoyeds layers to work with HF's forward signatures.
"""
import torch
import torch.nn as nn

# Import Samoyeds sparse layers
from module.linear.SSTransLinear import SSTransLinear
from module.linear.SSFusedSiluTransLinear import SSFusedSiluTransLinear
from module.linear.SPDenseWeightedTransLinear import SPDenseWeightedLinear
from module.linear.SPDenseTransLinear import SPDenseTransLinear


class HFCompatibleSSFusedSiluTransLinear(nn.Module):
    """Wrapper: HF calls forward(x), we call Samoyeds forward(x, idx)"""
    def __init__(self, samoyeds_layer):
        super().__init__()
        self.samoyeds_layer = samoyeds_layer
        
    def forward(self, x):
        # x: [batch_size, hidden_dim]
        batch_size = x.shape[0]
        idx = torch.arange(batch_size, device=x.device, dtype=torch.long)
        # Samoyeds forward returns [hidden_dim, padded_batch] transposed
        output = self.samoyeds_layer(x, idx)
        # output: [hidden_dim, padded_batch], take first batch_size and transpose back
        return output[:, :batch_size].T  # [batch_size, hidden_dim]


class HFCompatibleSSTransLinear(nn.Module):
    """Wrapper: HF calls forward(x), we call Samoyeds forward(x, idx)"""
    def __init__(self, samoyeds_layer):
        super().__init__()
        self.samoyeds_layer = samoyeds_layer
        
    def forward(self, x):
        batch_size = x.shape[0]
        idx = torch.arange(batch_size, device=x.device, dtype=torch.long)
        output = self.samoyeds_layer(x, idx)
        return output[:, :batch_size].T  # [batch_size, hidden_dim]


class HFCompatibleSPDenseWeightedLinear(nn.Module):
    """
    Wrapper: HF calls forward(x) then multiplies by routing_weights externally
    Samoyeds expects forward(x, routing_weights)
    
    We'll store routing_weights in a buffer and access during forward.
    """
    def __init__(self, samoyeds_layer):
        super().__init__()
        self.samoyeds_layer = samoyeds_layer
        self.routing_weights = None  # Set by parent MoE layer
        
    def forward(self, x):
        if self.routing_weights is None:
            # Fallback: uniform weights
            batch_size = x.shape[0]
            weights = torch.ones(batch_size, device=x.device, dtype=x.dtype)
        else:
            weights = self.routing_weights
            
        # Call Samoyeds with routing weights
        return self.samoyeds_layer(x, weights)


class HFCompatibleSPDenseTransLinear(nn.Module):
    """Wrapper: SPDenseTransLinear takes only input (no idx)"""
    def __init__(self, samoyeds_layer):
        super().__init__()
        self.samoyeds_layer = samoyeds_layer
        
    def forward(self, x):
        return self.samoyeds_layer(x)


class HFCompatibleQwen2MoeMLP(nn.Module):
    """
    Wrapper for entire expert MLP that handles Samoyeds sparse layers.
    Replaces HF's Qwen2MoeMLP forward.
    """
    def __init__(self, gate_proj, up_proj, down_proj, act_fn):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn
        
    def forward(self, x):
        # Gate path (includes SiLU in SSFusedSiluTransLinear)
        gate_out = self.gate_proj(x)  # Already has SiLU fused
        
        # Up path
        up_out = self.up_proj(x)
        
        # Element-wise multiply and down proj
        return self.down_proj(gate_out * up_out)