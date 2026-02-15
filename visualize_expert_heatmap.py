#!/usr/bin/env python3
"""
Generate expert activation heatmap from profiling data
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def apply_plot_style(font_family='serif', font_name='STIXGeneral', font_scale=1.0):
    """Apply publication-oriented typography with user-configurable scaling."""
    base = {
        'font.size': 24,
        'axes.labelsize': 28,
        'axes.titlesize': 30,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 24,
        'figure.titlesize': 32,
    }

    plt.rcParams['font.family'] = font_family
    if font_name:
        if font_family == 'serif':
            plt.rcParams['font.serif'] = [font_name]
        elif font_family == 'sans-serif':
            plt.rcParams['font.sans-serif'] = [font_name]

    for key, value in base.items():
        plt.rcParams[key] = value * font_scale

def load_profiling_data(json_path):
    """Load expert profiling data from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def extract_activation_matrix(data, layer_indices=None, expert_indices=None):
    """Extract activation rates as a matrix [layers x experts]
    
    Args:
        data: Profiling data dictionary
        layer_indices: List of layer indices to include (None = all)
        expert_indices: List of expert indices to include (None = all)
    """
    num_layers = len(data['layers'])
    num_experts = data['num_experts']
    
    # Determine which layers and experts to include
    if layer_indices is None:
        layer_indices = list(range(num_layers))
    if expert_indices is None:
        expert_indices = list(range(num_experts))
    
    activation_matrix = np.zeros((len(layer_indices), len(expert_indices)))
    
    for i, layer_idx in enumerate(layer_indices):
        layer_key = str(layer_idx)
        if layer_key in data['layers']:
            expert_stats = data['layers'][layer_key]['expert_stats']
            for j, expert_id in enumerate(expert_indices):
                expert_key = str(expert_id)
                if expert_key in expert_stats:
                    activation_matrix[i, j] = expert_stats[expert_key]['activation_rate']
    
    return activation_matrix, layer_indices, expert_indices

def plot_heatmap(
    activation_matrix,
    model_name,
    output_path,
    layer_indices=None,
    expert_indices=None,
    auto_scale=False,
    rate_unit='raw',
    cmap_name='YlOrRd',
    cbar_labelpad=28,
    cbar_tick_pad=2,
):
    """Create and save heatmap visualization"""
    num_layers, num_experts = activation_matrix.shape
    
    # Use actual indices for tick labels
    if layer_indices is None:
        layer_indices = list(range(num_layers))
    if expert_indices is None:
        expert_indices = list(range(num_experts))
    
    # Create figure with appropriate size
    fig_width = max(12, num_experts * 0.15)
    fig_height = max(8, num_layers * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Colormap configurable from CLI for publication comparisons.
    cmap = sns.color_palette(cmap_name, as_cmap=True)
    
    # Determine color scale range
    vmin = activation_matrix.min() if auto_scale else 0
    vmax = activation_matrix.max()
    
    # Plot heatmap
    im = ax.imshow(activation_matrix, cmap=cmap, aspect='auto', 
                   interpolation='nearest', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    unit_label = {
        'raw': 'Activation Rate',
        'pct': 'Activation Rate (%)',
        'bps': 'Activation Rate (bps)',
    }[rate_unit]
    cbar.set_label(unit_label, rotation=270, labelpad=cbar_labelpad)
    cbar.ax.tick_params(pad=cbar_tick_pad)
    
    # Set axis labels
    ax.set_xlabel('Expert ID', fontweight='bold')
    ax.set_ylabel('Layer Index', fontweight='bold')
    # ax.set_title(f'Expert Activation Patterns - {model_name}', 
    #              fontweight='bold', pad=20)
    
    # Set ticks with actual layer/expert indices
    ax.set_xticks(np.arange(num_experts))
    ax.set_yticks(np.arange(num_layers))
    ax.set_xticklabels(expert_indices)
    ax.set_yticklabels(layer_indices)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add grid for better readability
    ax.set_xticks(np.arange(num_experts) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_layers) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save as PDF
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {output_path}")
    
    return fig

def print_statistics(activation_matrix, model_name, rate_unit='raw'):
    """Print summary statistics"""
    num_layers, num_experts = activation_matrix.shape
    
    print(f"\n{'='*70}")
    print(f"Expert Activation Analysis - {model_name}")
    print(f"{'='*70}")
    print(f"Model configuration:")
    print(f"  • Total experts: {num_experts}")
    print(f"  • Total layers: {num_layers}")
    suffix = {'raw': '', 'pct': '%', 'bps': ' bps'}[rate_unit]
    print(f"\nActivation statistics:")
    print(f"  • Mean activation rate: {activation_matrix.mean():.4f}{suffix}")
    print(f"  • Std activation rate: {activation_matrix.std():.4f}{suffix}")
    print(f"  • Min activation rate: {activation_matrix.min():.4f}{suffix}")
    print(f"  • Max activation rate: {activation_matrix.max():.4f}{suffix}")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize expert activation heatmap')
    parser.add_argument('--input', type=str, 
                       default='artifacts/exact_profiling/deepseek-moe-16b-base_real_e64.json',
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
    
    # Visualization options
    parser.add_argument('--auto-scale', action='store_true',
                       help='Scale colormap from min to max (better contrast per model, but not comparable across models)')
    parser.add_argument('--rate-unit', choices=['raw', 'pct', 'bps'], default='raw',
                       help='Display unit for activation rate in heatmap/statistics.')
    parser.add_argument('--font-family', choices=['serif', 'sans-serif'], default='serif',
                       help='Font family for figure text (default: serif)')
    parser.add_argument('--font-name', type=str, default='STIXGeneral',
                       help='Preferred font name within selected family (default: STIXGeneral)')
    parser.add_argument('--font-scale', type=float, default=1.15,
                       help='Global font-size multiplier (default: 1.15)')
    parser.add_argument('--cmap', type=str, default='YlOrRd',
                       help='Matplotlib/seaborn colormap name (e.g., YlOrRd, viridis, magma, cividis)')
    parser.add_argument('--cbar-labelpad', type=float, default=28.0,
                       help='Colorbar label padding in points (default: 28)')
    parser.add_argument('--cbar-tick-pad', type=float, default=2.0,
                       help='Colorbar tick-label padding in points (default: 2)')
    
    args = parser.parse_args()

    # Apply typography after argument parsing.
    apply_plot_style(args.font_family, args.font_name, args.font_scale)
    
    # Load data
    print(f"Loading profiling data from: {args.input}")
    data = load_profiling_data(args.input)
    
    # Extract model name from file path
    model_name = Path(args.input).stem
    
    # Generate output path if not provided
    if args.output is None:
        output_dir = Path('artifacts/exact_profiling/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f'{model_name}_heatmap.pdf'
    
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
    
    # Extract activation matrix
    activation_matrix, layer_indices, expert_indices = extract_activation_matrix(
        data, layer_indices, expert_indices)
    
    # Convert display unit for visualization/statistics.
    if args.rate_unit == 'pct':
        activation_matrix = activation_matrix * 100.0
    elif args.rate_unit == 'bps':
        activation_matrix = activation_matrix * 10000.0

    # Print statistics
    print_statistics(activation_matrix, model_name, args.rate_unit)
    if layer_indices and len(layer_indices) < num_layers:
        print(f"Displaying layers: {layer_indices[0]}-{layer_indices[-1]} (sampled: {len(layer_indices)}/{num_layers})")
    if expert_indices and len(expert_indices) < num_experts:
        print(f"Displaying experts: {expert_indices[0]}-{expert_indices[-1]} ({len(expert_indices)}/{num_experts})\n")
    
    # Create heatmap
    fig = plot_heatmap(
        activation_matrix,
        model_name,
        args.output,
        layer_indices,
        expert_indices,
        args.auto_scale,
        args.rate_unit,
        args.cmap,
        args.cbar_labelpad,
        args.cbar_tick_pad,
    )
    
    # Show plot if requested
    if args.show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == '__main__':
    main()
