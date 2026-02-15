import torch
from transformers import AutoModelForCausalLM
from safetensors import safe_open
import os

def load_pretrained_weights_into_samoyeds(samoyeds_model, hf_checkpoint_path):
    """
    Transfer HuggingFace checkpoint weights into Samoyeds Mixtral model structure
    
    Args:
        samoyeds_model: Your Samoyeds Mixtral model (SSMixtralForCausalLM, etc.)
        hf_checkpoint_path: Path to HF checkpoint or model name
    
    Returns:
        samoyeds_model with loaded weights
    """
    # print(f"Loading weights from {hf_checkpoint_path}...")
    
    # Load state dict only (memory efficient)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(hf_checkpoint_path, trust_remote_code=True)
    state_dict = {}
    
    # Load from safetensors shards
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(hf_checkpoint_path, allow_patterns=["*.safetensors", "*.bin"])
    
    import glob
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if safetensor_files:
        for file in sorted(safetensor_files):
            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    else:
        # Fallback to pickle files
        bin_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))
        for file in sorted(bin_files):
            state_dict.update(torch.load(file, map_location="cpu"))
    
    # Transfer weights layer by layer
    samoyeds_layers = samoyeds_model.layers if hasattr(samoyeds_model, 'layers') else samoyeds_model.model.layers
    num_layers = len(samoyeds_layers)

    for idx in range(num_layers):
        sam_layer = samoyeds_layers[idx]
        
        # Copy attention weights
        for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            key = f"model.layers.{idx}.self_attn.{name}.weight"
            if key in state_dict:
                getattr(sam_layer.self_attn, name).weight.data.copy_(state_dict[key])
        
        # Copy norm weights
        for norm_name in ['input_layernorm', 'post_attention_layernorm']:
            key = f"model.layers.{idx}.{norm_name}.weight"
            if key in state_dict:
                getattr(sam_layer, norm_name).weight.data.copy_(state_dict[key])
        
        # Copy MoE weights (Mixtral has no shared expert, only routed experts)
        if hasattr(sam_layer, 'block_sparse_moe'):
            sam_moe = sam_layer.block_sparse_moe
            
            # Gate
            gate_key = f"model.layers.{idx}.block_sparse_moe.gate.weight"
            if gate_key in state_dict:
                sam_moe.gate.weight.data.copy_(state_dict[gate_key])
            
            # Routed experts (Mixtral uses w1, w2, w3 naming)
            num_experts = len(sam_moe.experts)
            for exp_idx in range(num_experts):
                for w_name in ['w1', 'w2', 'w3']:
                    key = f"model.layers.{idx}.block_sparse_moe.experts.{exp_idx}.{w_name}.weight"
                    if key in state_dict:
                        getattr(sam_moe.experts[exp_idx], w_name).weight.data.copy_(state_dict[key])
        
        if (idx + 1) % 5 == 0:
            print(f"  Loaded {idx + 1}/{num_layers} layers")
    
    # Copy embedding and output layers
    base_model = samoyeds_model.model if hasattr(samoyeds_model, 'model') else samoyeds_model
    
    if hasattr(base_model, 'embed_tokens') and 'model.embed_tokens.weight' in state_dict:
        base_model.embed_tokens.weight.data.copy_(state_dict['model.embed_tokens.weight'])
    if hasattr(base_model, 'norm') and 'model.norm.weight' in state_dict:
        base_model.norm.weight.data.copy_(state_dict['model.norm.weight'])
    
    # Copy lm_head
    if hasattr(samoyeds_model, 'lm_head') and 'lm_head.weight' in state_dict:
        samoyeds_model.lm_head.weight.data.copy_(state_dict['lm_head.weight'])
    
    # Clean up
    del state_dict
    import gc
    gc.collect()
    
    # print("✓ Weights loaded successfully")
    return samoyeds_model
