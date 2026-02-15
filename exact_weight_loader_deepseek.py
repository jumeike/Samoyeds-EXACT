import torch
from transformers import AutoModelForCausalLM

def load_pretrained_weights_into_samoyeds(samoyeds_model, hf_checkpoint_path):
    """
    Transfer HuggingFace checkpoint weights into Samoyeds DeepSeek model structure
    
    Args:
        samoyeds_model: Your Samoyeds DeepSeek model (SSDeepseekForCausalLM, etc.)
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

    # Debug: basic model stats
    # print(f"DEBUG: HF has {len(hf_layers)} layers, Samoyeds has {len(samoyeds_layers)} layers")
    # if len(hf_layers) > 0 and len(samoyeds_layers) > 0:
    #     try:
            # print(f"DEBUG: HF layer 0 attn q_proj shape: {hf_layers[0].self_attn.q_proj.weight.shape}")
            # print(f"DEBUG: Samoyeds layer 0 attn q_proj shape BEFORE: {samoyeds_layers[0].self_attn.q_proj.weight.shape}")
            # print(f"DEBUG: HF q_proj mean: {hf_layers[0].self_attn.q_proj.weight.mean().item():.6f}")
            # print(f"DEBUG: Samoyeds q_proj mean BEFORE: {samoyeds_layers[0].self_attn.q_proj.weight.mean().item():.6f}")
        # except Exception as e:
            # print(f"DEBUG: Attn shape/mean debug failed: {e}")

    for idx, (hf_layer, sam_layer) in enumerate(zip(hf_layers, samoyeds_layers)):
        # Copy attention weights
        if idx == 0:
            try:
                before_qproj = sam_layer.self_attn.q_proj.weight.data.clone()
                before_qproj_mean = before_qproj.mean().item()
            except Exception as e:
                before_qproj = None
                before_qproj_mean = float('nan')
                # print(f"DEBUG: Could not snapshot attn q_proj before copy: {e}")

        sam_layer.self_attn.load_state_dict(hf_layer.self_attn.state_dict(), strict=False)

        if idx == 0:
            try:
                after_qproj = sam_layer.self_attn.q_proj.weight.data
                after_qproj_mean = after_qproj.mean().item()
                eq_hf_qproj = torch.allclose(after_qproj, hf_layer.self_attn.q_proj.weight.data, rtol=1e-5, atol=1e-6)
                changed_from_before = None
                if before_qproj is not None:
                    changed_from_before = (after_qproj - before_qproj).abs().max().item()
                # print(f"DEBUG: Attn q_proj mean BEFORE: {before_qproj_mean:.6f}")
                # print(f"DEBUG: Attn q_proj mean AFTER:  {after_qproj_mean:.6f}")
                # print(f"DEBUG: Attn q_proj equal to HF:  {eq_hf_qproj}")
                # Strict bitwise equality check to mirror Qwen's signal
                strict_equal = torch.equal(after_qproj, hf_layer.self_attn.q_proj.weight.data)
                # print(f"DEBUG: Weight changed: {not strict_equal}")
                # if changed_from_before is not None:
                    # print(f"DEBUG: Attn q_proj changed_from_before (max abs diff): {changed_from_before:.6f}")
            except Exception as e:
                print(f"DEBUG: Attn after-copy debug failed: {e}")
        
        # Copy norm weights
        sam_layer.input_layernorm.load_state_dict(hf_layer.input_layernorm.state_dict())
        sam_layer.post_attention_layernorm.load_state_dict(hf_layer.post_attention_layernorm.state_dict())
        
        # Copy MoE weights (DeepSeek has shared experts)
        if hasattr(hf_layer, 'mlp') and hasattr(hf_layer.mlp, 'experts'):
            hf_moe = hf_layer.mlp
            sam_moe = sam_layer.mlp
            
            # Gate (MoEGate has .weight directly, not .gate.weight)
            if idx == 0:
                try:
                    before_exp0_gate = sam_moe.experts[0].gate_proj.weight.data.clone()
                    before_shared_gate = sam_moe.shared_experts.gate_proj.weight.data.clone() if hasattr(sam_moe, 'shared_experts') else None
                    # print(f"DEBUG: Samoyeds expert0 gate_proj mean BEFORE: {before_exp0_gate.mean().item():.6f}")
                    # if before_shared_gate is not None:
                        # print(f"DEBUG: Samoyeds shared_experts gate_proj mean BEFORE: {before_shared_gate.mean().item():.6f}")
                except Exception as e:
                    print(f"DEBUG: MoE before-copy snapshot failed: {e}")
            sam_moe.gate.weight.data.copy_(hf_moe.gate.weight.data)
            
            # Shared experts (DeepSeek specific)
            if hasattr(hf_moe, 'shared_experts') and hasattr(sam_moe, 'shared_experts'):
                sam_moe.shared_experts.gate_proj.weight.data.copy_(hf_moe.shared_experts.gate_proj.weight.data)
                sam_moe.shared_experts.up_proj.weight.data.copy_(hf_moe.shared_experts.up_proj.weight.data)
                sam_moe.shared_experts.down_proj.weight.data.copy_(hf_moe.shared_experts.down_proj.weight.data)
            
            # Debug: first MoE layer stats
            # if idx == 0:
            #     try:
            #         # print(f"DEBUG: HF MoE experts: {len(hf_moe.experts)}; Samoyeds MoE experts: {len(sam_moe.experts)}")
            #         # print(f"DEBUG: HF gate mean: {hf_moe.gate.weight.mean().item():.6f}")
            #         # After copying routed expert 0, check diffs
            #         # print(f"DEBUG: Samoyeds expert0 gate_proj mean AFTER: {sam_moe.experts[0].gate_proj.weight.mean().item():.6f}")
            #         eq_exp0_gate = torch.allclose(sam_moe.experts[0].gate_proj.weight.data, hf_moe.experts[0].gate_proj.weight.data, rtol=1e-5, atol=1e-6)
            #         # print(f"DEBUG: Expert0 gate_proj equal to HF: {eq_exp0_gate}")
            #         if 'before_exp0_gate' in locals() and before_exp0_gate is not None:
            #             diff_exp0 = (sam_moe.experts[0].gate_proj.weight.data - before_exp0_gate).abs().max().item()
            #             # print(f"DEBUG: Expert0 gate_proj changed_from_before (max abs diff): {diff_exp0:.6f}")
            #         if hasattr(hf_moe, 'shared_experts') and hasattr(sam_moe, 'shared_experts'):
            #             # print(f"DEBUG: Samoyeds shared_experts gate_proj mean AFTER: {sam_moe.shared_experts.gate_proj.weight.mean().item():.6f}")
            #             eq_shared_gate = torch.allclose(sam_moe.shared_experts.gate_proj.weight.data, hf_moe.shared_experts.gate_proj.weight.data, rtol=1e-5, atol=1e-6)
            #             # print(f"DEBUG: Shared_experts gate_proj equal to HF: {eq_shared_gate}")
            #             if 'before_shared_gate' in locals() and before_shared_gate is not None:
            #                 diff_shared = (sam_moe.shared_experts.gate_proj.weight.data - before_shared_gate).abs().max().item()
            #                 # print(f"DEBUG: Shared_experts gate_proj changed_from_before (max abs diff): {diff_shared:.6f}")
            #     except Exception as e:
            #         print(f"DEBUG: MoE debug failed: {e}")

            # Routed experts
            for exp_idx, (hf_expert, sam_expert) in enumerate(zip(hf_moe.experts, sam_moe.experts)):
                sam_expert.gate_proj.weight.data.copy_(hf_expert.gate_proj.weight.data)
                sam_expert.up_proj.weight.data.copy_(hf_expert.up_proj.weight.data)
                sam_expert.down_proj.weight.data.copy_(hf_expert.down_proj.weight.data)
        
        if (idx + 1) % 5 == 0:
            print(f"  Loaded {idx + 1}/{len(hf_layers)} layers")
    
    # Copy embedding and output layers
    base_model = samoyeds_model.model if hasattr(samoyeds_model, 'model') else samoyeds_model
    
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
