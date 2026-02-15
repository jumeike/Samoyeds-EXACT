#!/usr/bin/env python3
"""
Generate stacked bar chart showing hot/cold expert distribution across layers
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Set style for publication-quality plots (matching visualize_expert_heatmap.py)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 32

def load_profiling_data(json_path):
    """Load expert profiling data from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_hot_cold_counts_per_layer(data, layer_indices=None, expert_indices=None):
    """Extract hot/cold expert counts from per-layer classification
    
    Returns:
        hot_counts: list of hot expert counts per layer
        cold_counts: list of cold expert counts per layer
        layer_indices: actual layer indices used
    """
    num_layers = len(data['layers'])
    num_experts = data['num_experts']
    
    if layer_indices is None:
        layer_indices = list(range(num_layers))
    if expert_indices is None:
        expert_indices = list(range(num_experts))
    
    expert_indices_set = set(expert_indices)
    hot_counts = []
    cold_counts = []
    
    for layer_idx in layer_indices:
        layer_key = str(layer_idx)
        if layer_key not in data['layers']:
            hot_counts.append(0)
            cold_counts.append(0)
            continue
        
        layer_data = data['layers'][layer_key]
        
        # Use pre-computed classification if available
        if 'classification' in layer_data:
            hot_experts = set(layer_data['classification']['hot'])
            cold_experts = set(layer_data['classification']['cold'])
            
            # Filter by expert_indices if specified
            hot = len(hot_experts & expert_indices_set)
            cold = len(cold_experts & expert_indices_set)
        else:
            # Fallback: no classification available
            hot = 0
            cold = 0
        
        hot_counts.append(hot)
        cold_counts.append(cold)
    
    return hot_counts, cold_counts, layer_indices

def plot_stacked_bar(hot_counts, cold_counts, layer_indices, model_name, output_path):
    """Create and save stacked bar chart"""
    num_layers = len(layer_indices)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(14, num_layers * 0.6), 10))
    
    # Create stacked bar chart
    x = np.arange(num_layers)
    width = 0.8
    
    # Plot bars
    p1 = ax.bar(x, hot_counts, width, label='Hot Experts', color='#bb2b5f', edgecolor='black', linewidth=1.5)
    p2 = ax.bar(x, cold_counts, width, bottom=hot_counts, label='Cold Experts', color='#4e95d9', edgecolor='black', linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Number of Experts', fontweight='bold')
    # ax.set_title(f'Hot/Cold Expert Distribution - {model_name}', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_indices)
    legend = ax.legend(loc='upper right')
    for text in legend.get_texts():
        text.set_fontstyle('italic')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels on bars (optional, only if not too many layers)
    if num_layers <= 32:
        for i, (hot, cold) in enumerate(zip(hot_counts, cold_counts)):
            if hot > 0:
                ax.text(i, hot/2, str(hot), ha='center', va='center', 
                       color='white', fontweight='bold', fontsize=26)
            if cold > 0:
                ax.text(i, hot + cold/2, str(cold), ha='center', va='center',
                       color='white', fontweight='bold', fontsize=26)
    
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Bar chart saved to: {output_path}")
    
    return fig

def print_statistics(hot_counts, cold_counts, layer_indices, model_name):
    """Print summary statistics"""
    total_hot = sum(hot_counts)
    total_cold = sum(cold_counts)
    total_experts = total_hot + total_cold
    
    print(f"\n{'='*70}")
    print(f"Hot/Cold Expert Distribution - {model_name}")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  • Layers analyzed: {len(layer_indices)}")
    print(f"\nOverall statistics:")
    print(f"  • Total hot experts: {total_hot} ({100*total_hot/total_experts:.1f}%)")
    print(f"  • Total cold experts: {total_cold} ({100*total_cold/total_experts:.1f}%)")
    print(f"\nPer-layer statistics:")
    print(f"  • Avg hot per layer: {np.mean(hot_counts):.1f}")
    print(f"  • Avg cold per layer: {np.mean(cold_counts):.1f}")
    print(f"  • Min hot: {min(hot_counts)}, Max hot: {max(hot_counts)}")
    print(f"  • Min cold: {min(cold_counts)}, Max cold: {max(cold_counts)}")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize hot/cold expert distribution')
    parser.add_argument('--input', type=str, 
                       default='artifacts/exact_profiling/deepseek-moe-16b-base_real_e64_layer.json',
                       help='Path to profiling JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PDF path (default: auto-generated)')
    parser.add_argument('--show', action='store_true',
                       help='Display plot interactively')
    
    # Subsetting options
    parser.add_argument('--max-layers', type=int, default=None,
                       help='Show only first N layers')
    parser.add_argument('--layer-range', type=str, default=None,
                       help='Layer range as start:end (e.g., 0:10)')
    parser.add_argument('--sample-layers', type=int, default=None,
                       help='Evenly sample N layers across all')
    parser.add_argument('--expert-range', type=str, default=None,
                       help='Expert range as start:end (e.g., 0:32)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading profiling data from: {args.input}")
    data = load_profiling_data(args.input)
    
    # Extract model name from file path
    model_name = Path(args.input).stem
    
    # Generate output path if not provided
    if args.output is None:
        output_dir = Path('artifacts/exact_profiling/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f'{model_name}_distribution.pdf'
    
    # Determine layer and expert indices
    num_layers = len(data['layers'])
    num_experts = data['num_experts']
    
    layer_indices = None
    expert_indices = None
    
    # Process layer subsetting options
    if args.max_layers:
        layer_indices = list(range(min(args.max_layers, num_layers)))
    elif args.layer_range:
        start, end = map(int, args.layer_range.split(':'))
        layer_indices = list(range(start, min(end, num_layers)))
    elif args.sample_layers:
        step = max(1, num_layers // args.sample_layers)
        layer_indices = list(range(0, num_layers, step))[:args.sample_layers]
    
    # Process expert subsetting
    if args.expert_range:
        start, end = map(int, args.expert_range.split(':'))
        expert_indices = list(range(start, min(end, num_experts)))
    
    # Get hot/cold counts per layer from pre-computed classification
    hot_counts, cold_counts, layer_indices = get_hot_cold_counts_per_layer(
        data, layer_indices, expert_indices)
    
    # Print statistics
    print_statistics(hot_counts, cold_counts, layer_indices, model_name)
    
    if layer_indices and len(layer_indices) < num_layers:
        print(f"Displaying layers: {layer_indices[0]}-{layer_indices[-1]} (sampled: {len(layer_indices)}/{num_layers})")
    if expert_indices and len(expert_indices) < num_experts:
        print(f"Displaying experts: {expert_indices[0]}-{expert_indices[-1]} ({len(expert_indices)}/{num_experts})\n")
    
    # Create bar chart
    fig = plot_stacked_bar(hot_counts, cold_counts, layer_indices, model_name, args.output)
    
    # Show plot if requested
    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == '__main__':
    main()
