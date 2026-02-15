"""
Calibration-based one-shot pruning for EXACT
Prunes based on weight salience (|weight| × |activation|)
"""
import torch
import torch.nn as nn
from typing import Dict, List
import json
from tqdm import tqdm

class ActivationCollector:
    """Collects expert activations during forward pass"""
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self, model, moe_layers):
        """Register hooks on expert inputs"""
        for layer_idx, moe_layer in enumerate(moe_layers):
            for expert_idx, expert in enumerate(moe_layer.experts):
                # Find first linear layer (gate_proj or w1)
                linear = None
                if hasattr(expert, 'gate_proj'):
                    linear = expert.gate_proj
                elif hasattr(expert, 'w1'):
                    linear = expert.w1
                
                if linear:
                    hook = linear.register_forward_hook(
                        self._make_hook(layer_idx, expert_idx)
                    )
                    self.hooks.append(hook)
    
    def _make_hook(self, layer_idx, expert_idx):
        def hook(module, input, output):
            key = (layer_idx, expert_idx)
            if key not in self.activations:
                self.activations[key] = []
            # Store input activation magnitude per hidden dim
            act = input[0].detach().float().abs().mean(dim=(0, 1))  # [hidden_dim]
            self.activations[key].append(act.cpu())
        return hook
    
    def get_average_activations(self):
        """Average activations across samples"""
        avg_acts = {}
        for key, acts in self.activations.items():
            avg_acts[key] = torch.stack(acts).mean(dim=0)
        return avg_acts
    
    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}


def compute_salience(weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    """
    Compute salience: |weight| × |activation|
    
    Args:
        weight: [out_features, in_features]
        activation: [in_features]
    Returns:
        salience: [out_features, in_features]
    """
    # Broadcast activation to match weight shape
    act_expanded = activation.unsqueeze(0).expand_as(weight)
    salience = weight.abs() * act_expanded
    return salience


def prune_by_salience(weight: torch.Tensor, salience: torch.Tensor, 
                      n: int, m: int) -> torch.Tensor:
    """
    Prune to N:M pattern based on salience scores
    
    Args:
        weight: [rows, cols]
        salience: [rows, cols]
        n: keep N elements
        m: per group of M
    Returns:
        pruned_weight: same shape
    """
    nrows, ncols = weight.shape
    assert ncols % m == 0, f"cols {ncols} not divisible by M={m}"
    
    # Reshape to [rows, groups, m]
    weight_groups = weight.view(nrows, ncols // m, m)
    salience_groups = salience.view(nrows, ncols // m, m)
    
    # Find top-N per group based on salience
    _, topn_idx = salience_groups.topk(n, dim=2)
    
    # Create mask and apply
    mask = torch.zeros_like(weight_groups, dtype=torch.bool)
    mask.scatter_(2, topn_idx, True)
    pruned = torch.where(mask, weight_groups, torch.zeros_like(weight_groups))
    
    return pruned.view(nrows, ncols)


# def collect_calibration_activations(model, tokenizer, calib_texts: List[str], 
#                                     moe_layers, device='cuda') -> Dict:
#     """Collect activations on calibration set"""
#     collector = ActivationCollector()
#     collector.register_hooks(model, moe_layers)
    
#     model.eval()
#     print(f"Collecting activations on {len(calib_texts)} samples...")
    
#     with torch.no_grad():
#         for text in tqdm(calib_texts, desc="Calibration"):
#             inputs = tokenizer(text, return_tensors='pt', max_length=512, 
#                              truncation=True).to(device)
#             model(**inputs)
    
#     avg_acts = collector.get_average_activations()
#     collector.clear()
#     return avg_acts

def collect_calibration_activations(model, tokenizer, calib_texts: List[str], 
                                    moe_layers, device='cuda') -> Dict:
    collector = ActivationCollector()
    collector.register_hooks(model, moe_layers)
    
    model.eval()
    print(f"Collecting activations on {len(calib_texts)} samples...")
    
    with torch.no_grad():
        for text in tqdm(calib_texts, desc="Calibration"):
            inputs = tokenizer(text, return_tensors='pt', max_length=512, 
                             truncation=True)
            # Move to model's first device instead of hardcoded device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)
    
    avg_acts = collector.get_average_activations()
    collector.clear()
    return avg_acts


def apply_calibrated_pruning(model, profiling_json: str, 
                             calib_activations: Dict) -> nn.Module:
    """
    Apply EXACT pruning with calibration-based weight selection
    
    Args:
        model: Model with MoE layers
        profiling_json: Path to hot/cold classification
        calib_activations: Dict[(layer_idx, expert_idx)] -> activation tensor
    Returns:
        model with pruned weights (in-place)
    """
    # Load classification
    with open(profiling_json) as f:
        profile = json.load(f)
    
    # Support both formats
    if 'layer_classifications' in profile:
        layer_classifications = profile['layer_classifications']
    elif 'layers' in profile:
        # Extract from nested structure
        layer_classifications = {}
        for layer_idx, layer_data in profile['layers'].items():
            if 'classification' in layer_data:
                layer_classifications[layer_idx] = layer_data['classification']
    else:
        raise ValueError("Unknown profiling JSON format")
    
    # Get MoE layers
    moe_layers = get_moe_layers(model)
    
    print(f"Applying calibrated pruning to {len(moe_layers)} MoE layers...")
    
    for layer_idx_str, classification in layer_classifications.items():
        layer_idx = int(layer_idx_str)
        moe = moe_layers[layer_idx]
        
        # Prune hot experts (2:4)
        for expert_idx in classification['hot']:
            expert = moe.experts[expert_idx]
            act = calib_activations.get((layer_idx, expert_idx))
            
            if act is None:
                print(f"  Warning: No activation for layer {layer_idx} expert {expert_idx}, using magnitude")
                act = torch.ones(expert.gate_proj.weight.shape[1] if hasattr(expert, 'gate_proj') 
                               else expert.w1.weight.shape[1])
            
            prune_expert_weights(expert, act, n=2, m=4)
        
        # Prune cold experts (1:4)
        for expert_idx in classification['cold']:
            expert = moe.experts[expert_idx]
            act = calib_activations.get((layer_idx, expert_idx))
            
            if act is None:
                act = torch.ones(expert.gate_proj.weight.shape[1] if hasattr(expert, 'gate_proj') 
                               else expert.w1.weight.shape[1])
            
            prune_expert_weights(expert, act, n=1, m=4)
        
        if (layer_idx + 1) % 5 == 0:
            print(f"  Processed layers 0-{layer_idx}")
    
    print("Calibrated pruning complete")
    return model


def prune_expert_weights(expert, activation: torch.Tensor, n: int, m: int):
    """Prune all weights in expert using salience"""
    # Determine weight names (support both Mixtral and DeepSeek/Qwen)
    if hasattr(expert, 'w1'):
        weight_names = [('w1', True), ('w3', True), ('w2', False)]
    else:
        weight_names = [('gate_proj', True), ('up_proj', True), ('down_proj', False)]
    
    for name, use_activation in weight_names:
        if not hasattr(expert, name):
            continue
            
        linear = getattr(expert, name)
        weight = linear.weight.data.float().cpu()
        
        # Compute salience
        if use_activation:
            salience = compute_salience(weight, activation.cpu())
        else:
            # Output projection: use magnitude only
            salience = weight.abs()
        
        # Prune
        pruned = prune_by_salience(weight, salience, n, m)
        linear.weight.data = pruned.to(linear.weight.device).half()

def fine_tune_pruned(model, tokenizer, calib_texts: List[str], 
                     steps: int = 100, lr: float = 1e-6, device='cuda'):  # Lower LR
    from torch.optim import AdamW
    
    # Only tune non-zero weights
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    optimizer = AdamW(params, lr=lr, weight_decay=0.01)  # Add weight decay
    model.train()
    
    print(f"Fine-tuning for {steps} steps (lr={lr})...")
    step = 0
    
    pbar = tqdm(total=steps, desc="Fine-tuning")
    while step < steps:
        for text in calib_texts[:128]:  # Reduce samples
            if step >= steps:
                break
            
            inputs = tokenizer(text, return_tensors='pt', max_length=256,
                             truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            if torch.isnan(loss):
                continue  # Skip NaN batches
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            step += 1
    
    pbar.close()
    model.eval()

# def fine_tune_pruned(model, tokenizer, calib_texts: List[str], 
#                      steps: int = 100, lr: float = 1e-5, device='cuda'):
#     """Fine-tune pruned model"""
#     from torch.optim import AdamW
    
#     optimizer = AdamW(model.parameters(), lr=lr)
#     model.train()
    
#     print(f"Fine-tuning for {steps} steps (lr={lr})...")
#     step = 0
    
#     pbar = tqdm(total=steps, desc="Fine-tuning")
#     while step < steps:
#         for text in calib_texts:
#             if step >= steps:
#                 break
            
#             inputs = tokenizer(text, return_tensors='pt', max_length=512,
#                              truncation=True).to(device)
#             outputs = model(**inputs, labels=inputs['input_ids'])
#             loss = outputs.loss
            
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
            
#             pbar.update(1)
#             pbar.set_postfix({'loss': f'{loss.item():.4f}'})
#             step += 1
    
#     pbar.close()
#     model.eval()

# In exact_calibration.py, replace fine_tune_pruned():
# def fine_tune_pruned(model, tokenizer, calib_texts: List[str], 
#                      steps: int = 100, lr: float = 5e-6, device='cuda'):
#     from torch.optim import SGD  # Changed from AdamW
    
#     optimizer = SGD(model.parameters(), lr=lr)  # No momentum buffers
#     model.train()
    
#     print(f"Fine-tuning for {steps} steps (lr={lr})...")
#     step = 0
    
#     pbar = tqdm(total=steps, desc="Fine-tuning")
#     while step < steps:
#         for text in calib_texts[:64]:  # Reduce samples
#             if step >= steps:
#                 break
            
#             inputs = tokenizer(text, return_tensors='pt', max_length=256,  # Shorter context
#                              truncation=True).to(device)
#             outputs = model(**inputs, labels=inputs['input_ids'])
#             loss = outputs.loss
            
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
            
#             pbar.update(1)
#             pbar.set_postfix({'loss': f'{loss.item():.4f}'})
#             step += 1
    
#     pbar.close()
#     model.eval()

def get_moe_layers(model):
    """Extract MoE layers from model"""
    moe_layers = []
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            if hasattr(layer, 'block_sparse_moe'):
                moe_layers.append(layer.block_sparse_moe)
            elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                moe_layers.append(layer.mlp)
    
    return moe_layers
