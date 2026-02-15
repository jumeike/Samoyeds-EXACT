#!/usr/bin/env python3
"""
Measure GPU memory usage for model implementations.
Returns peak memory allocated in MB.
"""

import argparse
import sys
import torch

def measure_memory(model_script, batch_size, expert, intermediate_size, hidden_size, seq_len):
    """Run a model script and measure peak memory usage."""
    import subprocess
    import re
    
    # Build the command
    cmd = [
        'python', model_script,
        '--batch_size', str(batch_size),
        '--layer',
        '--flash',
        '--experts', str(expert),
        '--hidden_size', str(hidden_size),
        '--intermediate_size', str(intermediate_size),
        '--seq_len', str(seq_len)
    ]
    
    # Run and capture output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Look for memory usage in output (we'll need to add this to the model scripts)
        # For now, return a placeholder
        return 0.0
        
    except subprocess.TimeoutExpired:
        print(f"Timeout running {model_script}", file=sys.stderr)
        return -1
    except Exception as e:
        print(f"Error running {model_script}: {e}", file=sys.stderr)
        return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_script', required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--expert', type=int, required=True)
    parser.add_argument('--intermediate_size', type=int, required=True)
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    
    args = parser.parse_args()
    
    memory_mb = measure_memory(
        args.model_script,
        args.batch_size,
        args.expert,
        args.intermediate_size,
        args.hidden_size,
        args.seq_len
    )
    
    print(memory_mb)
