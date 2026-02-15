import torch
from transformers import AutoModelForCausalLM

def load_pretrained_weights_into_samoyeds(samoyeds_model, hf_checkpoint_path):
    """
    Transfer HuggingFace checkpoint weights into Samoyeds model structure
    
    Args:
        samoyeds_model: Your Samoyeds MoE model (SSQwen2MoeModel, etc.)
        hf_checkpoint_path: Path to HF checkpoint or model name
    
    Returns:
        samoyeds_model with loaded weights
    """
    # print(f"Loading weights from {hf_checkpoint_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_checkpoint_path,
        torch_dtype=torch.float16,
        device_map='cpu',
        trust_remote_code=True
    )
    
    # Transfer weights layer by layer
    hf_layers = hf_model.model.layers
    samoyeds_layers = samoyeds_model.layers if hasattr(samoyeds_model, 'layers') else samoyeds_model.model.layers
    
    # print(f"DEBUG: HF has {len(hf_layers)} layers, Samoyeds has {len(samoyeds_layers)} layers")
    # print(f"DEBUG: HF layer 0 attn weight shape: {hf_layers[23].self_attn.q_proj.weight.shape}")
    # print(f"DEBUG: Samoyeds layer 0 attn weight shape BEFORE: {samoyeds_layers[23].self_attn.q_proj.weight.shape}")
    # print(f"DEBUG: HF layer 0 attn keys: {list(hf_layers[23].self_attn.state_dict().keys())[:5]}")
    # print(f"DEBUG: Samoyeds layer 0 attn keys: {list(samoyeds_layers[23].self_attn.state_dict().keys())[:5]}")
    # print(f"DEBUG: HF q_proj mean: {hf_layers[23].self_attn.q_proj.weight.mean().item():.6f}")
    # print(f"DEBUG: Samoyeds q_proj mean BEFORE: {samoyeds_layers[23].self_attn.q_proj.weight.mean().item():.6f}")
    # print(f"DEBUG: HF device: {hf_layers[23].self_attn.q_proj.weight.device}")
    # print(f"DEBUG: Samoyeds device: {samoyeds_layers[23].self_attn.q_proj.weight.device}")
    
    for idx, (hf_layer, sam_layer) in enumerate(zip(hf_layers, samoyeds_layers)):
        # Copy attention weights - use direct copy instead of load_state_dict
        sam_layer.self_attn.q_proj.weight.data.copy_(hf_layer.self_attn.q_proj.weight.data)
        sam_layer.self_attn.k_proj.weight.data.copy_(hf_layer.self_attn.k_proj.weight.data)
        sam_layer.self_attn.v_proj.weight.data.copy_(hf_layer.self_attn.v_proj.weight.data)
        sam_layer.self_attn.o_proj.weight.data.copy_(hf_layer.self_attn.o_proj.weight.data)
        if hasattr(hf_layer.self_attn.q_proj, 'bias') and hf_layer.self_attn.q_proj.bias is not None:
            sam_layer.self_attn.q_proj.bias.data.copy_(hf_layer.self_attn.q_proj.bias.data)
            sam_layer.self_attn.k_proj.bias.data.copy_(hf_layer.self_attn.k_proj.bias.data)
            sam_layer.self_attn.v_proj.bias.data.copy_(hf_layer.self_attn.v_proj.bias.data)
        
        # Copy norm weights
        sam_layer.input_layernorm.load_state_dict(hf_layer.input_layernorm.state_dict())
        sam_layer.post_attention_layernorm.load_state_dict(hf_layer.post_attention_layernorm.state_dict())
        
        # Copy MoE weights
        if hasattr(hf_layer, 'mlp') and hasattr(hf_layer.mlp, 'experts'):
            # MoE layer
            hf_moe = hf_layer.mlp
            sam_moe = sam_layer.mlp
            
            # if idx == 0:
                # print(f"DEBUG: Layer {idx} has MoE, copying {len(hf_moe.experts)} experts")
                # print(f"DEBUG: HF expert 59 gate_proj type: {type(hf_moe.experts[59].gate_proj)}")
                # print(f"DEBUG: Samoyeds expert 59 gate_proj type: {type(sam_moe.experts[59].gate_proj)}")
                # print(f"DEBUG: HF expert 59 gate_proj mean: {hf_moe.experts[59].gate_proj.weight.mean().item():.6f}")
                # print(f"DEBUG: Samoyeds expert 59 gate_proj mean BEFORE: {sam_moe.experts[59].gate_proj.weight.mean().item():.6f}")
            
            # Gate
            sam_moe.gate.weight.data.copy_(hf_moe.gate.weight.data)
            
            # Shared expert (Qwen2-MoE specific)
            if hasattr(hf_moe, 'shared_expert'):
                sam_moe.shared_expert.gate_proj.weight.data.copy_(hf_moe.shared_expert.gate_proj.weight.data)
                sam_moe.shared_expert.up_proj.weight.data.copy_(hf_moe.shared_expert.up_proj.weight.data)
                sam_moe.shared_expert.down_proj.weight.data.copy_(hf_moe.shared_expert.down_proj.weight.data)
            
            # Routed experts
            for exp_idx, (hf_expert, sam_expert) in enumerate(zip(hf_moe.experts, sam_moe.experts)):
                # Check if Samoyeds expert has .weight directly or wrapped
                if hasattr(sam_expert.gate_proj, 'weight'):
                    sam_expert.gate_proj.weight.data.copy_(hf_expert.gate_proj.weight.data)
                    sam_expert.up_proj.weight.data.copy_(hf_expert.up_proj.weight.data)
                    sam_expert.down_proj.weight.data.copy_(hf_expert.down_proj.weight.data)
                elif hasattr(sam_expert.gate_proj, 'linear'):
                    # Wrapped in custom layer like SSFusedSiluTransLinear
                    sam_expert.gate_proj.linear.weight.data.copy_(hf_expert.gate_proj.weight.data)
                    sam_expert.up_proj.linear.weight.data.copy_(hf_expert.up_proj.weight.data)
                    sam_expert.down_proj.linear.weight.data.copy_(hf_expert.down_proj.weight.data)
                else:
                    raise AttributeError(f"Cannot find weight tensor in Samoyeds expert {exp_idx}")
            
            # if idx == 0:
                # print(f"DEBUG: Samoyeds expert 59 gate_proj mean AFTER: {sam_moe.experts[59].gate_proj.weight.mean().item():.6f}")
                # Check if shared expert also copied
                # if hasattr(sam_moe, 'shared_expert'):
                    # print(f"DEBUG: Samoyeds shared expert gate_proj mean AFTER: {sam_moe.shared_expert.gate_proj.weight.mean().item():.6f}")
        elif hasattr(hf_layer, 'mlp'):
            # Dense FFN layer (non-MoE)
            hf_mlp = hf_layer.mlp
            sam_mlp = sam_layer.mlp
            
            # if idx == 1:  # Layer 1 is typically non-MoE
                # print(f"DEBUG: Layer {idx} has dense FFN, copying weights")
            
            sam_mlp.gate_proj.weight.data.copy_(hf_mlp.gate_proj.weight.data)
            sam_mlp.up_proj.weight.data.copy_(hf_mlp.up_proj.weight.data)
            sam_mlp.down_proj.weight.data.copy_(hf_mlp.down_proj.weight.data)
        
        if (idx + 1) % 5 == 0:
            print(f"  Loaded {idx + 1}/{len(hf_layers)} layers")
    
    # print(f"DEBUG: Samoyeds layer 0 attn weight shape AFTER: {samoyeds_layers[23].self_attn.q_proj.weight.shape}")
    # print(f"DEBUG: Samoyeds q_proj mean AFTER: {samoyeds_layers[23].self_attn.q_proj.weight.mean().item():.6f}")
    # print(f"DEBUG: Weight changed: {not torch.equal(samoyeds_layers[23].self_attn.q_proj.weight, hf_layers[23].self_attn.q_proj.weight)}")
    
    # Copy embedding and output layers
    base_model = samoyeds_model.model if hasattr(samoyeds_model, 'model') else samoyeds_model
    
    # print(f"DEBUG: Copying embeddings and LM head...")
    # print(f"DEBUG: Has embed_tokens? {hasattr(base_model, 'embed_tokens')}")
    # print(f"DEBUG: Has norm? {hasattr(base_model, 'norm')}")
    # print(f"DEBUG: Has lm_head? {hasattr(samoyeds_model, 'lm_head') and hasattr(hf_model, 'lm_head')}")
    
    if hasattr(base_model, 'embed_tokens'):
        # print(f"DEBUG: HF embed_tokens mean: {hf_model.model.embed_tokens.weight.mean().item():.6f}")
        # print(f"DEBUG: Samoyeds embed_tokens mean BEFORE: {base_model.embed_tokens.weight.mean().item():.6f}")
        base_model.embed_tokens.weight.data.copy_(hf_model.model.embed_tokens.weight.data)
        # print(f"DEBUG: Samoyeds embed_tokens mean AFTER: {base_model.embed_tokens.weight.mean().item():.6f}")
    if hasattr(base_model, 'norm'):
        base_model.norm.load_state_dict(hf_model.model.norm.state_dict())
    
    # Copy LM head if present (for ForCausalLM models)
    if hasattr(samoyeds_model, 'lm_head') and hasattr(hf_model, 'lm_head'):
        # print(f"DEBUG: HF lm_head mean: {hf_model.lm_head.weight.mean().item():.6f}")
        # print(f"DEBUG: Samoyeds lm_head mean BEFORE: {samoyeds_model.lm_head.weight.mean().item():.6f}")
        samoyeds_model.lm_head.weight.data.copy_(hf_model.lm_head.weight.data)
        # print(f"DEBUG: Samoyeds lm_head mean AFTER: {samoyeds_model.lm_head.weight.mean().item():.6f}")
    
    # print("✓ Weights loaded successfully")
    del hf_model
    torch.cuda.empty_cache()
    
    return samoyeds_model
