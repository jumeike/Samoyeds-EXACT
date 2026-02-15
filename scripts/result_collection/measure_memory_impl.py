#!/usr/bin/env python3
"""
Measure GPU memory usage for a specific model implementation.
"""

import argparse
import sys
import torch

def get_model_classes(model_name):
    """Import and return model classes based on model name."""
    if model_name == "mixtral":
        from mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralDecoderLayer
        from mixtral.configuration_mixtral import MixtralConfig
        from mixtral.modeling_mixtral_ss import SSMixtralDecoderLayer
        from mixtral.modeling_mixtral_exact import sparsemoeblock_to_exact
        return MixtralConfig, MixtralDecoderLayer, SSMixtralDecoderLayer
    elif model_name == "deepseek":
        from deepseek.modeling_deepseek import DeepseekV2DecoderLayer
        from deepseek.configuration_deepseek import DeepseekV2Config
        from deepseek.modeling_deepseek_ss import SSDeepseekV2DecoderLayer
        return DeepseekV2Config, DeepseekV2DecoderLayer, SSDeepseekV2DecoderLayer
    elif model_name == "qwen2_moe":
        from qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer
        from qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
        from qwen2_moe.modeling_qwen2_moe_ss import SSQwen2MoeDecoderLayer
        return Qwen2MoeConfig, Qwen2MoeDecoderLayer, SSQwen2MoeDecoderLayer
    else:
        raise ValueError(f"Unknown model: {model_name}")

def measure_memory_dense(model_name, batch_size, num_experts, intermediate_size, hidden_size, seq_len):
    """Measure memory for dense/transformers implementation."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    Config, DecoderLayer, _ = get_model_classes(model_name)
    
    config = Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
    )
    config._attn_implementation = 'flash_attention_2'
    
    layer = DecoderLayer(config, layer_idx=0).half().cuda()
    input_tensor = torch.rand((batch_size, seq_len, hidden_size)).half().cuda()
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).cuda()
    
    with torch.no_grad():
        if model_name == 'qwen2_moe':
            _ = layer(input_tensor, position_ids=position_ids)
        else:
            _ = layer(input_tensor)
    
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return memory_mb

def measure_memory_samoyeds(model_name, batch_size, num_experts, intermediate_size, hidden_size, seq_len):
    """Measure memory for Samoyeds implementation."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    Config, _, SSDecoderLayer = get_model_classes(model_name)
    
    config = Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_experts,
    )
    config._attn_implementation = 'flash_attention_2'
    
    layer = SSDecoderLayer(config, layer_idx=0).half().cuda()
    input_tensor = torch.rand((batch_size, seq_len, hidden_size)).half().cuda()
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).cuda()
    
    with torch.no_grad():
        if model_name == 'qwen2_moe':
            _ = layer(input_tensor, position_ids=position_ids)
        else:
            _ = layer(input_tensor)
    
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return memory_mb

def measure_memory_exact(model_name, batch_size, num_experts, intermediate_size, hidden_size, seq_len):
    """Measure memory for EXACT implementation."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    if model_name == "mixtral":
        from mixtral.modeling_mixtral import MixtralDecoderLayer
        from mixtral.configuration_mixtral import MixtralConfig
        from mixtral.modeling_mixtral_exact import decoderlayer_to_exact
        
        config = MixtralConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_local_experts=num_experts,
        )
        config._attn_implementation = 'flash_attention_2'
        
        layer = MixtralDecoderLayer(config, layer_idx=0)
        profiling_json = 'artifacts/exact_profiling/mixtral_e8.json'
        layer = decoderlayer_to_exact(layer, profiling_json)
        
    elif model_name == "deepseek":
        from deepseek.modeling_deepseek import DeepseekV2DecoderLayer
        from deepseek.configuration_deepseek import DeepseekV2Config
        from deepseek.modeling_deepseek_exact import decoderlayer_to_exact
        
        config = DeepseekV2Config(
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            n_routed_experts=num_experts,
        )
        config._attn_implementation = 'flash_attention_2'
        
        layer = DeepseekV2DecoderLayer(config, layer_idx=0)
        profiling_json = 'artifacts/exact_profiling/deepseek_e64.json'
        layer = decoderlayer_to_exact(layer, profiling_json)
        
    elif model_name == "qwen2_moe":
        from qwen2_moe.modeling_qwen2_moe import Qwen2MoeDecoderLayer
        from qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
        from qwen2_moe.modeling_qwen2_moe_exact import decoderlayer_to_exact
        
        config = Qwen2MoeConfig(
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            num_experts=num_experts,
        )
        config._attn_implementation = 'flash_attention_2'
        
        layer = Qwen2MoeDecoderLayer(config, layer_idx=0)
        profiling_json = 'artifacts/exact_profiling/qwen2_e60.json'
        layer = decoderlayer_to_exact(layer, profiling_json)
    
    layer = layer.half().cuda()
    input_tensor = torch.rand((batch_size, seq_len, hidden_size)).half().cuda()
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).cuda()
    
    with torch.no_grad():
        if model_name == 'qwen2_moe':
            _ = layer(input_tensor, position_ids=position_ids)
        else:
            _ = layer(input_tensor)
    
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return memory_mb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['mixtral', 'deepseek', 'qwen2_moe'])
    parser.add_argument('--implementation', required=True, choices=['dense', 'samoyeds', 'exact'])
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--num_experts', type=int, required=True)
    parser.add_argument('--intermediate_size', type=int, required=True)
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    
    args = parser.parse_args()
    
    try:
        if args.implementation == 'dense':
            memory_mb = measure_memory_dense(
                args.model, args.batch_size, args.num_experts,
                args.intermediate_size, args.hidden_size, args.seq_len
            )
        elif args.implementation == 'samoyeds':
            memory_mb = measure_memory_samoyeds(
                args.model, args.batch_size, args.num_experts,
                args.intermediate_size, args.hidden_size, args.seq_len
            )
        elif args.implementation == 'exact':
            memory_mb = measure_memory_exact(
                args.model, args.batch_size, args.num_experts,
                args.intermediate_size, args.hidden_size, args.seq_len
            )
        
        print(f"{memory_mb:.2f}")
    except Exception as e:
        print(f"0.00", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
