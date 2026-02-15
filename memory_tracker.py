"""
Memory Usage Tracker for EXACT
Tracks GPU memory during model loading and inference
"""
import torch
from contextlib import contextmanager
from typing import Dict, Optional
import psutil
import os


class MemoryTracker:
    """Track GPU and CPU memory usage"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all tracked metrics"""
        self.metrics = {
            'model_loading': {},
            'conversion': {},
            'exact_compression': {},
            'inference': {}
        }
        torch.cuda.reset_peak_memory_stats()
        
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage in GB"""
        if not torch.cuda.is_available():
            return {'allocated': 0, 'reserved': 0, 'peak': 0}
        
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'peak': torch.cuda.max_memory_allocated() / 1024**3    # GB
        }
    
    def get_cpu_memory(self) -> Dict[str, float]:
        """Get current CPU memory usage in GB"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            'rss': mem_info.rss / 1024**3,  # Resident Set Size in GB
            'vms': mem_info.vms / 1024**3   # Virtual Memory Size in GB
        }
    
    @contextmanager
    def track(self, phase: str):
        """Context manager to track memory for a specific phase
        
        Usage:
            with memory_tracker.track('model_loading'):
                model = load_model()
        """
        # Reset peak stats at start
        torch.cuda.reset_peak_memory_stats()
        
        # Record initial memory
        initial_gpu = self.get_gpu_memory()
        initial_cpu = self.get_cpu_memory()
        
        print(f"\n[Memory] {phase} - Initial:")
        print(f"  GPU: {initial_gpu['allocated']:.2f} GB allocated, {initial_gpu['reserved']:.2f} GB reserved")
        print(f"  CPU: {initial_cpu['rss']:.2f} GB RSS")
        
        try:
            yield self
        finally:
            # Record final memory
            final_gpu = self.get_gpu_memory()
            final_cpu = self.get_cpu_memory()
            
            # Calculate deltas
            delta_gpu = final_gpu['allocated'] - initial_gpu['allocated']
            delta_cpu = final_cpu['rss'] - initial_cpu['rss']
            
            # Store metrics
            self.metrics[phase] = {
                'initial_gpu_allocated': initial_gpu['allocated'],
                'final_gpu_allocated': final_gpu['allocated'],
                'peak_gpu_allocated': final_gpu['peak'],
                'delta_gpu': delta_gpu,
                'initial_cpu_rss': initial_cpu['rss'],
                'final_cpu_rss': final_cpu['rss'],
                'delta_cpu': delta_cpu
            }
            
            print(f"[Memory] {phase} - Final:")
            print(f"  GPU: {final_gpu['allocated']:.2f} GB allocated, {final_gpu['peak']:.2f} GB peak")
            print(f"  Delta: +{delta_gpu:+.2f} GB GPU, +{delta_cpu:+.2f} GB CPU")
    
    def print_summary(self):
        """Print comprehensive memory usage summary"""
        print("\n" + "="*80)
        print("MEMORY USAGE SUMMARY")
        print("="*80)
        
        total_peak = 0
        for phase, metrics in self.metrics.items():
            if metrics:
                print(f"\n{phase.upper().replace('_', ' ')}:")
                print(f"  GPU Allocated: {metrics['initial_gpu_allocated']:.2f} GB → {metrics['final_gpu_allocated']:.2f} GB")
                print(f"  GPU Peak: {metrics['peak_gpu_allocated']:.2f} GB")
                print(f"  GPU Delta: {metrics['delta_gpu']:+.2f} GB")
                print(f"  CPU RSS: {metrics['initial_cpu_rss']:.2f} GB → {metrics['final_cpu_rss']:.2f} GB")
                print(f"  CPU Delta: {metrics['delta_cpu']:+.2f} GB")
                total_peak = max(total_peak, metrics['peak_gpu_allocated'])
        
        print(f"\nOVERALL PEAK GPU MEMORY: {total_peak:.2f} GB")
        print("="*80)
    
    def get_summary_dict(self) -> Dict:
        """Get summary as dictionary for CSV export"""
        summary = {}
        for phase, metrics in self.metrics.items():
            if metrics:
                summary[f'{phase}_peak_gpu_gb'] = metrics['peak_gpu_allocated']
                summary[f'{phase}_delta_gpu_gb'] = metrics['delta_gpu']
                summary[f'{phase}_final_cpu_rss_gb'] = metrics['final_cpu_rss']
        
        # Overall peak
        if self.metrics:
            all_peaks = [m['peak_gpu_allocated'] for m in self.metrics.values() if m]
            summary['overall_peak_gpu_gb'] = max(all_peaks) if all_peaks else 0
        
        return summary


def measure_model_memory(model) -> Dict[str, float]:
    """Measure memory footprint of a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict with memory metrics in GB
    """
    param_memory = 0
    buffer_memory = 0
    
    # Count parameters
    for param in model.parameters():
        param_memory += param.nelement() * param.element_size()
    
    # Count buffers
    for buffer in model.buffers():
        buffer_memory += buffer.nelement() * buffer.element_size()
    
    total_memory = (param_memory + buffer_memory) / 1024**3  # Convert to GB
    
    return {
        'parameters_gb': param_memory / 1024**3,
        'buffers_gb': buffer_memory / 1024**3,
        'total_gb': total_memory,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_buffers': sum(b.numel() for b in model.buffers())
    }


def compare_model_sizes(baseline_model, exact_model) -> Dict:
    """Compare memory footprint of baseline vs EXACT compressed model
    
    Args:
        baseline_model: Dense baseline model
        exact_model: EXACT compressed model
        
    Returns:
        Dict with comparison metrics
    """
    baseline_mem = measure_model_memory(baseline_model)
    exact_mem = measure_model_memory(exact_model)
    
    compression_ratio = baseline_mem['total_gb'] / exact_mem['total_gb'] if exact_mem['total_gb'] > 0 else 0
    memory_saved = baseline_mem['total_gb'] - exact_mem['total_gb']
    
    print("\n" + "="*80)
    print("MODEL SIZE COMPARISON")
    print("="*80)
    print(f"\nBaseline Model:")
    print(f"  Parameters: {baseline_mem['num_parameters']:,} ({baseline_mem['parameters_gb']:.3f} GB)")
    print(f"  Buffers: {baseline_mem['num_buffers']:,} ({baseline_mem['buffers_gb']:.3f} GB)")
    print(f"  Total: {baseline_mem['total_gb']:.3f} GB")
    
    print(f"\nEXACT Compressed Model:")
    print(f"  Parameters: {exact_mem['num_parameters']:,} ({exact_mem['parameters_gb']:.3f} GB)")
    print(f"  Buffers: {exact_mem['num_buffers']:,} ({exact_mem['buffers_gb']:.3f} GB)")
    print(f"  Total: {exact_mem['total_gb']:.3f} GB")
    
    print(f"\nCompression:")
    print(f"  Ratio: {compression_ratio:.2f}x")
    print(f"  Memory Saved: {memory_saved:.3f} GB ({(memory_saved/baseline_mem['total_gb']*100):.1f}%)")
    print("="*80)
    
    return {
        'baseline_model_gb': baseline_mem['total_gb'],
        'exact_model_gb': exact_mem['total_gb'],
        'compression_ratio': compression_ratio,
        'memory_saved_gb': memory_saved,
        'memory_saved_percent': (memory_saved/baseline_mem['total_gb']*100) if baseline_mem['total_gb'] > 0 else 0
    }
