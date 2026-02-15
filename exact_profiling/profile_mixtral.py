#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
EXACT Profiling for Mixtral
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from mixtral.modeling_mixtral import MixtralSparseMoeBlock
from mixtral.configuration_mixtral import MixtralConfig
from exact_core.profiler import ExpertProfiler
from exact_core.exact_config import EXACTConfig


def main(args):
    # Mixtral configuration
    config = MixtralConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_local_experts=args.experts,
        num_experts_per_tok=2,
    )
    
    # Create MoE block
    print("Initializing Mixtral MoE block...")
    moe_block = MixtralSparseMoeBlock(config).half().cuda()
    moe_block.eval()
    
    # Create profiler
    exact_config = EXACTConfig(hot_threshold=args.hot_threshold)
    profiler = ExpertProfiler(
        moe_block=moe_block,
        num_experts=args.experts,
        model_name="mixtral",
        config=exact_config
    )
    
    # Run profiling
    results = profiler.profile(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        iterations=args.iterations,
        imbalance_ratio=args.imbalance_ratio,
        top_k=2
    )
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Profile Mixtral experts')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=4096)
    parser.add_argument('--intermediate_size', type=int, default=14336)
    parser.add_argument('--imbalance_ratio', type=float, default=1.5)
    parser.add_argument('--hot_threshold', type=float, default=0.1)
    
    args = parser.parse_args()
    main(args)
