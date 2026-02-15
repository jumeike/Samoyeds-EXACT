#!/usr/bin/env python3
"""
Verify EXACT adaptive compression produces correct outputs
Compares EXACT vs Samoyeds baseline (both sparse) at MoE block level
"""

import torch
import argparse
import numpy as np
from copy import deepcopy
from mixtral.modeling_mixtral_samoyeds import sparsemoeblock_to_ss
from mixtral.modeling_mixtral_exact import sparsemoeblock_to_exact
from transformers import AutoModelForCausalLM


def load_and_convert_both(checkpoint_path, profiling_json):
    """Load checkpoint once, convert MoE block to both Samoyeds and EXACT"""
    print("Loading checkpoint...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map='cpu'
    )
    
    dense_moe_block = model.model.layers[0].block_sparse_moe
    
    # Keep dense copy for comparison
    dense_moe = deepcopy(dense_moe_block)
    
    print("Converting to Samoyeds...")
    samoyeds_moe = sparsemoeblock_to_ss(dense_moe_block)
    
    print("Converting to EXACT...")
    exact_moe = sparsemoeblock_to_exact(dense_moe_block, profiling_json)
    
    del model
    return dense_moe, samoyeds_moe, exact_moe


def compare_outputs(out1, out2, rtol=1e-2, atol=1e-3):
    """Compare two outputs"""
    np1 = out1[0].detach().cpu().float().numpy()
    np2 = out2[0].detach().cpu().float().numpy()
    
    assert np1.shape == np2.shape, f"Shape mismatch: {np1.shape} vs {np2.shape}"
    
    abs_diff = np.abs(np1 - np2)
    rel_diff = abs_diff / (np.abs(np1) + 1e-8)
    
    close = np.allclose(np1, np2, rtol=rtol, atol=atol)
    
    print(f"Max abs diff: {abs_diff.max():.6f}, Max rel diff: {rel_diff.max():.4%}")
    print(f"Mean abs diff: {abs_diff.mean():.6f}, Mean rel diff: {rel_diff.mean():.4%}")
    print(f"Match: {'✓ PASS' if close else '✗ FAIL'}")
    
    return close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/mixtral-hf',
                       help='HuggingFace checkpoint path')
    parser.add_argument('--profiling-json', type=str,
                       default='artifacts/exact_profiling/mixtral_e8.json',
                       help='Expert profiling JSON')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seq-len', type=int, default=4096)
    parser.add_argument('--hidden-size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create test input
    print(f"\nGenerating test input: [{args.batch_size}, {args.seq_len}, {args.hidden_size}]")
    test_input = torch.randn(
        args.batch_size, args.seq_len, args.hidden_size,
        dtype=torch.bfloat16, device='cpu'
    )
    
    # Load once and convert to both
    print("\n" + "="*70)
    print("Loading checkpoint and converting...")
    print("="*70)
    dense_moe, samoyeds_moe, exact_moe = load_and_convert_both(args.checkpoint, args.profiling_json)
    
    print(f"\nDense type: {type(dense_moe).__name__}")
    print(f"Samoyeds type: {type(samoyeds_moe).__name__}")
    print(f"EXACT type: {type(exact_moe).__name__}")
    
    # print("\nRunning dense inference...")
    # with torch.no_grad():
    #     dense_output = dense_moe(test_input)
    
    print("\nRunning Samoyeds inference...")
    with torch.no_grad():
        samoyeds_output = samoyeds_moe(test_input)
    
    print("Running EXACT inference...")
    with torch.no_grad():
        exact_output = exact_moe(test_input)
    
    # Compare Samoyeds vs Dense
    # print("\n" + "="*70)
    # print("Samoyeds vs Dense Baseline")
    # print("="*70)
    # compare_outputs(dense_output, samoyeds_output)
    
    # Compare EXACT vs Samoyeds
    print("\n" + "="*70)
    print("EXACT vs Samoyeds Baseline")
    print("="*70)
    is_correct = compare_outputs(samoyeds_output, exact_output)
    
    print(f"\n{'='*70}")
    if is_correct:
        print("✓ EXACT matches Samoyeds baseline!")
    else:
        print("✗ WARNING: EXACT differs from Samoyeds")
    print(f"{'='*70}\n")
    
    return 0 if is_correct else 1


if __name__ == '__main__':
    exit(main())
