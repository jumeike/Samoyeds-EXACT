#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
EXACT-Compressed Mixtral Benchmark
Tests EXACT adaptive compression on Mixtral MoE
"""

import argparse
import time
import torch

from mixtral.modeling_mixtral import MixtralModel, MixtralSparseMoeBlock, MixtralDecoderLayer
from mixtral.configuration_mixtral import MixtralConfig
from mixtral.modeling_mixtral_exact import sparsemoeblock_to_exact

parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--time', action='store_true', default=False)

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=4096)

parser.add_argument('--mlp', action='store_true', default=False)
parser.add_argument('--layer', action='store_true', default=False)
parser.add_argument('--model', action='store_true', default=False)

parser.add_argument('--hidden_size', type=int, default=4096)
parser.add_argument('--intermediate_size', type=int, default=14336)

parser.add_argument('--experts', type=int, default=8)

parser.add_argument('--flash', action='store_true', default=False)
parser.add_argument('--trace_small', action='store_true', default=False,
                    help='Use tiny run setup for tracing/debug (noisy kernel/sparsifier debug).')

# EXACT-specific arguments
parser.add_argument('--profiling_json', type=str, default='artifacts/exact_profiling/mixtral-8x7b-v0.1_real_e8_layer.json',
                    help='Path to expert classification JSON')

args = parser.parse_args()

m = args.intermediate_size
k = args.hidden_size
n = args.batch_size * args.seq_len
expert_num = args.experts
use_flash = args.flash

WARMUP = 10
ITER = 100

if args.trace_small:
    # Keep model dimensions unchanged to avoid invalid kernel configs,
    # but reduce runtime workload to make tracing practical.
    args.batch_size = 1
    args.seq_len = 64
    WARMUP = 0
    ITER = 1

# Setup Mixtral configuration
configuration = MixtralConfig(
    vocab_size=32000,
    hidden_size=k,
    intermediate_size=m,
    num_hidden_layers=1,
    num_attention_heads=32,
    num_key_value_heads=8,
    hidden_act="silu",
    max_position_embeddings=4096 * 32,
    initializer_range=0.02,
    rms_norm_eps=1e-5,
    use_cache=True,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    tie_word_embeddings=False,
    rope_theta=1e6,
    sliding_window=4096,
    attention_dropout=0.0,
    num_experts_per_tok=2,
    num_local_experts=expert_num,
    output_router_logits=False,
    router_aux_loss_coef=0.001,
)

position_ids = None
if use_flash:
    configuration._attn_implementation = "flash_attention_2"
    position_ids = torch.arange(args.seq_len).unsqueeze(0).expand(args.batch_size, args.seq_len).cuda()
else:
    configuration._attn_implementation = "eager"


def mixtral_mlp_run():
    """Test EXACT-compressed MoE block"""
    dense_model = MixtralSparseMoeBlock(configuration)
    exact_model = sparsemoeblock_to_exact(dense_model, args.profiling_json)
    exact_model = exact_model.half().cuda()
    exact_model.eval()

    # # VERIFY ADAPTIVE COMPRESSION
    # print("\n" + "="*70)
    # print("EXACT COMPRESSION VERIFICATION")
    # print("="*70)
    
    total_params = 0
    hot_params = 0
    cold_params = 0
    
    # Load classification to determine hot/cold
    # import json
    # with open(args.profiling_json, 'r') as f:
    #     classification_data = json.load(f)

    #  # Get classification (handle both formats)
    # if 'layers' in classification_data:
    #     # Per-layer format - use layer 0 for single MoE benchmark
    #     layer_data = classification_data['layers'].get('0', {})
    #     layer_classification = layer_data.get('classification', {})
    #     hot_experts = layer_classification.get('hot', [])
    #     cold_experts = layer_classification.get('cold', [])
    # else:
    #     # Old global format (fallback)
    #     hot_experts = classification_data.get('hot_experts', [])
    #     cold_experts = classification_data.get('cold_experts', [])
    
    # # TRUE COMPRESSION CALCULATION
    # hot_count = len(hot_experts)
    # cold_count = args.experts - hot_count 
    # dense_per_expert = (14336 * 4096 * 2) + (4096 * 14336)

    # hot_compressed = hot_count * dense_per_expert * 0.5
    # cold_compressed = cold_count * dense_per_expert * 0.25
    # exact_total_calc = hot_compressed + cold_compressed
    # dense_total = args.experts * dense_per_expert
    # true_ratio = dense_total / exact_total_calc

    # print(f"\nTRUE COMPRESSION ANALYSIS:")
    # print(f"  Dense: {dense_total:,} params")
    # print(f"  Hot ({hot_count}×50%): {int(hot_compressed):,} params")
    # print(f"  Cold ({cold_count}×25%): {int(cold_compressed):,} params")
    # print(f"  EXACT total: {int(exact_total_calc):,} params")
    # print(f"  TRUE compression: {true_ratio:.2f}x ({100*(1-1/true_ratio):.1f}% saved)")

    # Input shape: (batch_size, sequence_length, hidden_size)
    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            final_hidden_states, router_logits = exact_model(input)
        torch.cuda.synchronize()
        end = time.time()
        print("Mixtral,mlp,EXACT,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size, 
               args.experts, (end - start)*1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/Mixtral_EXACT_MLP_'+torch.cuda.get_device_name().split(' ')[1].split('-')[0]+'_'+str(args.batch_size)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            final_hidden_states, router_logits = exact_model(input)
            prof.step()

        prof.stop()


def mixtral_decoder_layer_run():
    """Test EXACT-compressed decoder layer"""
    # print("[!] Layer mode: Converting MoE block within decoder layer to EXACT")
    
    # Create decoder layer with dense MoE
    layer = MixtralDecoderLayer(configuration, 0)
    
    # Convert the MoE block to EXACT
    layer.block_sparse_moe = sparsemoeblock_to_exact(layer.block_sparse_moe, args.profiling_json)
    
    exact_model = layer.half().cuda()
    exact_model.eval()

    # Input shape: (batch_size, sequence_length, hidden_size)
    input = torch.rand((args.batch_size, args.seq_len, k)).half().cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                torch.cuda.synchronize()
                start = time.time()
            output, = exact_model(input, position_ids=position_ids)
        torch.cuda.synchronize()
        end = time.time()
        print("Mixtral,layer,EXACT,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size,
               args.experts, (end - start)*1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=10, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/Mixtral_EXACT_Layer_'+torch.cuda.get_device_name().split(' ')[1].split('-')[0]+'_'+str(args.batch_size)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            output, = exact_model(input, position_ids=position_ids)
            prof.step()

        prof.stop()


def mixtral_model_run():
    """Test EXACT-compressed full 32-layer model"""
    # print("[!] Model mode: Converting all MoE blocks in 32-layer model to EXACT")
    
    configuration.num_hidden_layers = 32
    model = MixtralModel(configuration)
    
    # Convert all MoE blocks to EXACT
    for layer in model.layers:
        layer.block_sparse_moe = sparsemoeblock_to_exact(layer.block_sparse_moe, args.profiling_json)
    
    exact_model = model.half().cuda()
    exact_model.eval()

    # Input shape: (batch_size, sequence_length)
    input = torch.randint(low=0, high=32000, size=(args.batch_size, args.seq_len)).cuda()

    if args.time:
        for i in range(ITER + WARMUP):
            if i == WARMUP:
                start = time.time()
            output = exact_model(input)
        end = time.time()
        print("Mixtral,model,EXACT,%d,%d,%d,%d,%d,%d,%s,%s" %
              (ITER, args.batch_size, args.seq_len, args.hidden_size, args.intermediate_size,
               args.experts, (end - start)*1000, configuration._attn_implementation))

    if args.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=3, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './outputs/profiler/Mixtral_EXACT_model_'+torch.cuda.get_device_name().split(' ')[1].split('-')[0]),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()

        for i in range(ITER):
            output = exact_model(input)
            prof.step()

        prof.stop()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    torch.set_grad_enabled(False)

    # print('# EXACT Benchmark: model,model type,kernel type,iter,batch_size,seq_len,hidden_size,intermediate_size,expert_num,time,atten_mode')
    # print(f'# Profiling JSON: {args.profiling_json}')

    if args.mlp:
        mixtral_mlp_run()
    if args.layer:
        mixtral_decoder_layer_run()
    if args.model:
        mixtral_model_run()
