import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read memory usage data
results_dir = Path('artifacts/results/model_accuracy_results')

# Use most recent memory files
deepseek_memory_files = sorted(results_dir.glob('memory_usage_deepseek-ai*.csv'))
qwen_memory_files = sorted(results_dir.glob('memory_usage_Qwen*.csv'))

if not deepseek_memory_files or not qwen_memory_files:
    print("Error: Could not find memory usage CSV files for both models")
    exit(1)

deepseek_df = pd.read_csv(deepseek_memory_files[-1])
qwen_df = pd.read_csv(qwen_memory_files[-1])

print(f"Using DeepSeek data from: {deepseek_memory_files[-1].name}")
print(f"  Task: {deepseek_df['tasks'].values[0]}")
print(f"Using Qwen data from: {qwen_memory_files[-1].name}")
print(f"  Task: {qwen_df['tasks'].values[0]}")

# Extract data
models = ['DeepSeek-MoE-16B', 'Qwen1.5-MoE-A2.7B']
baseline_memory = [
    deepseek_df['baseline_inference_peak_gpu_gb'].values[0],
    qwen_df['baseline_inference_peak_gpu_gb'].values[0]
]
exact_memory = [
    deepseek_df['exact_inference_peak_gpu_gb'].values[0],
    qwen_df['exact_inference_peak_gpu_gb'].values[0]
]
compression_ratios = [
    deepseek_df['compression_ratio'].values[0],
    qwen_df['compression_ratio'].values[0]
]
memory_saved_percent = [
    deepseek_df['memory_saved_percent'].values[0],
    qwen_df['memory_saved_percent'].values[0]
]

# Create output directory
output_dir = Path('artifacts/results/memory_plots')
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.rcParams['font.size'] = 28
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['xtick.labelsize'] = 32
plt.rcParams['ytick.labelsize'] = 32
plt.rcParams['legend.fontsize'] = 30

# Define color scheme (matching visualize_throughput.py)
colors = {
    'baseline': '#C73E1D',  # Red (GEMM color)
    'exact': '#2E86AB'      # Blue (EXACT color)
}

# Plot 1: Memory Savings (Baseline vs EXACT)
fig, ax = plt.subplots(figsize=(14, 10))

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_memory, width, label='Baseline', color=colors['baseline'], alpha=0.8)
bars2 = ax.bar(x + width/2, exact_memory, width, label='EXACT', color=colors['exact'], alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f} GB',
                ha='center', va='bottom', fontsize=28)

# Add percentage savings text between the bars
for i, (baseline, exact, saving) in enumerate(zip(baseline_memory, exact_memory, memory_saved_percent)):
    # Place text at midpoint between the two bars
    y_pos = (baseline + exact) / 2
    ax.text(i, y_pos, f'{saving:.1f}%\nsaved',
            ha='center', va='center', fontsize=28, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

ax.set_ylabel('Peak GPU Memory (GB)')
ax.set_xlabel('Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(output_dir / 'memory_savings.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'memory_savings.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'memory_savings.png'}")
print(f"Saved: {output_dir / 'memory_savings.pdf'}")

# Plot 2: Compression Ratio
fig, ax = plt.subplots(figsize=(14, 10))

x = np.arange(len(models))
bars = ax.bar(x, compression_ratios, color=['#C73E1D', '#F18F01'], alpha=0.8, width=0.5)

# Add value labels on bars
for bar, ratio in zip(bars, compression_ratios):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{ratio:.2f}×',
            ha='center', va='bottom', fontsize=32)

ax.set_ylabel('Compression Ratio')
ax.set_xlabel('Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='No compression')
ax.legend(loc='upper center')

plt.tight_layout()
plt.savefig(output_dir / 'compression_ratio.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'compression_ratio.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'compression_ratio.png'}")
print(f"Saved: {output_dir / 'compression_ratio.pdf'}")

# Print summary statistics
print("\n" + "="*70)
print("MEMORY SAVINGS SUMMARY")
print("="*70)
for model, baseline, exact, ratio, saving in zip(models, baseline_memory, exact_memory, 
                                                   compression_ratios, memory_saved_percent):
    print(f"\n{model}:")
    print(f"  Baseline Peak GPU: {baseline:.2f} GB")
    print(f"  EXACT Peak GPU: {exact:.2f} GB")
    print(f"  Memory Saved: {baseline - exact:.2f} GB ({saving:.1f}%)")
    print(f"  Compression Ratio: {ratio:.2f}×")
print("="*70)
