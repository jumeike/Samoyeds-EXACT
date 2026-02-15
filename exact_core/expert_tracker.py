#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
Expert Usage Tracking for EXACT
"""
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List
import json
from pathlib import Path


class ExpertUsageTracker:
    """Tracks expert activation patterns during inference"""
    
    def __init__(self, num_experts: int, num_layers: int = 1):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.enabled = False
        
        # Track counts per layer
        self.expert_counts = defaultdict(lambda: defaultdict(int))
        self.total_tokens = defaultdict(int)
        self.total_activations = 0
        
    def enable(self):
        self.enabled = True
        print("[EXACT] Expert tracking enabled")
        
    def disable(self):
        self.enabled = False
        
    def reset(self):
        self.expert_counts.clear()
        self.total_tokens.clear()
        self.total_activations = 0
        
    def log_routing(self, layer_idx: int, selected_experts: torch.Tensor):
        """Log expert selections from routing"""
        if not self.enabled:
            return

        if selected_experts.ndim == 3:
            # [batch, seq, topk] -> [tokens, topk]
            batch_size, seq_len, _ = selected_experts.shape
            batch_tokens = batch_size * seq_len
            selected_experts = selected_experts.reshape(-1, selected_experts.shape[-1])
        else:
            batch_tokens = selected_experts.shape[0]

        self.total_tokens[layer_idx] += batch_tokens

        expert_ids = selected_experts.flatten().cpu().numpy()
        for expert_id in expert_ids:
            self.expert_counts[layer_idx][int(expert_id)] += 1
            
        self.total_activations += len(expert_ids)
        
    def get_usage_stats(self, layer_idx: int = 0) -> Dict:
        """Get usage statistics for a layer"""
        if layer_idx not in self.expert_counts:
            return {}
            
        # Use total expert selections (not tokens) as denominator
        total_selections = sum(self.expert_counts[layer_idx].values())
        if total_selections == 0:
            total_selections = 1
        
        stats = {}
        for expert_id in range(self.num_experts):
            count = self.expert_counts[layer_idx].get(expert_id, 0)
            stats[expert_id] = {
                'count': count,
                'percentage': (count / total_selections) * 100,
                'activation_rate': count / total_selections
            }
            
        return stats

    def get_entropy(self, layer_idx: int = 0) -> Dict[str, float]:
        """Return entropy and normalized entropy for a layer's routing distribution."""
        stats = self.get_usage_stats(layer_idx)
        if not stats:
            return {"entropy": 0.0, "entropy_norm": 0.0}

        probs = []
        for expert_id in range(self.num_experts):
            probs.append(stats[expert_id]["activation_rate"])

        # Avoid log(0) by skipping zeros
        entropy = 0.0
        for p in probs:
            if p > 0.0:
                entropy -= p * np.log(p)

        max_entropy = np.log(self.num_experts) if self.num_experts > 1 else 1.0
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
        return {"entropy": float(entropy), "entropy_norm": float(entropy_norm)}
        
    def classify_experts(self, threshold: float = 0.1, layer_idx: int = 0) -> Dict[str, List[int]]:
        """Classify experts as hot or cold"""
        stats = self.get_usage_stats(layer_idx)
        
        hot_experts = []
        cold_experts = []
        
        for expert_id, data in stats.items():
            if data['activation_rate'] >= threshold:
                hot_experts.append(expert_id)
            else:
                cold_experts.append(expert_id)
                
        return {'hot': sorted(hot_experts), 'cold': sorted(cold_experts)}
        
    def print_summary(self, layer_idx: int = 0):
        """Print usage summary"""
        stats = self.get_usage_stats(layer_idx)
        ent = self.get_entropy(layer_idx)
        
        print(f"\n{'='*60}")
        print(f"EXACT Expert Usage Summary (Layer {layer_idx})")
        print(f"{'='*60}")
        print(f"Total tokens processed: {self.total_tokens[layer_idx]:,}")
        print(f"Entropy: {ent['entropy']:.4f} | Entropy_norm: {ent['entropy_norm']:.4f}")
        print(f"\nPer-Expert Statistics:")
        print(f"{'Expert':<10} {'Count':<12} {'%':<10} {'Bar'}")
        print(f"{'-'*60}")
        
        for expert_id in sorted(stats.keys()):
            data = stats[expert_id]
            bar_length = int(data['percentage'] / 2)
            bar = '█' * bar_length
            print(f"{expert_id:<10} {data['count']:<12,} {data['percentage']:<9.2f}% {bar}")
            
    def save_stats(self, output_path: str, layer_classifications: Dict[int, List[int]] = None):
        """Save statistics to JSON
        Args:
            layer_classifications: Dict mapping layer_idx -> {'hot': [...], 'cold': [...]}
                                If None, no classification saved (backward compatible)
        """
        data = {
            'num_experts': self.num_experts,
            'num_layers': self.num_layers,
            'layers': {}
        }
        
        for layer_idx in range(self.num_layers):
            stats = self.get_usage_stats(layer_idx)
            ent = self.get_entropy(layer_idx)
            layer_data = {
                'total_tokens': self.total_tokens[layer_idx],
                'expert_stats': stats,
                'entropy': ent["entropy"],
                'entropy_norm': ent["entropy_norm"],
            }

            # Add per-layer classification if provided
            if layer_classifications and layer_idx in layer_classifications:
                layer_data['classification'] = layer_classifications[layer_idx]
            
            data['layers'][str(layer_idx)] = layer_data
            
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"[EXACT] Stats saved to {output_path}")
