#
# Copyright (c) 2025 EXACT Project
# Licensed under the Apache License, Version 2.0
#

"""
EXACT Adaptive Sparsifier
Applies variable sparsity based on expert usage classification
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .exact_config import EXACTConfig


class EXACTSparsifier:
    """Adaptive sparsifier for MoE expert weights"""
    
    def __init__(self, config: Optional[EXACTConfig] = None):
        """
        Args:
            config: EXACT configuration
        """
        self.config = config or EXACTConfig()
        
    def load_classification(self, profile_path: str) -> Dict[str, List[int]]:
        """
        Load expert classification from profiling results
        
        Args:
            profile_path: Path to profiling JSON file
            
        Returns:
            Dict with 'hot' and 'cold' expert lists
        """
        with open(profile_path, 'r') as f:
            data = json.load(f)
        
        # Extract usage stats for layer 0
        stats = data['layers']['0']['expert_stats']
        total_selections = sum(stats[str(i)]['count'] for i in range(data['num_experts']))
        
        hot_experts = []
        cold_experts = []
        
        for expert_id in range(data['num_experts']):
            activation_rate = stats[str(expert_id)]['count'] / total_selections
            if activation_rate >= self.config.hot_threshold:
                hot_experts.append(expert_id)
            else:
                cold_experts.append(expert_id)
        
        return {'hot': sorted(hot_experts), 'cold': sorted(cold_experts)}
    
    def apply_2_4_sparsity(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Apply 2:4 structured sparsity via magnitude pruning
        
        Args:
            weight: Dense weight tensor of shape (out_features, in_features)
            
        Returns:
            Sparse weight tensor with 2:4 pattern
        """
        weight = weight.clone()
        
        # Reshape to process in groups of 4
        out_features, in_features = weight.shape
        assert in_features % 4 == 0, "Input features must be divisible by 4"
        
        # Reshape: (out_features, in_features // 4, 4)
        reshaped = weight.view(out_features, in_features // 4, 4)
        
        # Get magnitudes and find indices of 2 smallest in each group of 4
        magnitudes = reshaped.abs()
        _, indices = torch.topk(magnitudes, k=2, dim=2, largest=False)
        
        # Create mask and zero out 2 smallest values
        mask = torch.ones_like(reshaped)
        mask.scatter_(2, indices, 0)
        
        # Apply mask and reshape back
        sparse_weight = (reshaped * mask).view(out_features, in_features)
        
        return sparse_weight
    
    def apply_1_4_sparsity(self, weight: torch.Tensor) -> torch.Tensor:
        """
        TEMPORARY: Apply 2:4 sparsity for cold experts
        TODO: True 1:4 requires kernel modifications to replace SPTC_N/SPTC_M macros
        
        The Samoyeds kernel has hardcoded macros:
          #define SPTC_N 2
          #define SPTC_M 4
        These are used for ALL memory calculations, ignoring template parameters.
        Until we replace these with template N, M throughout the kernel,
        we must use 2:4 format for all experts.
        
        Args:
            weight: Dense weight tensor
            
        Returns:
            Sparse weight tensor with 2:4 pattern (same as hot for now)
        """
        # Use same 2:4 pattern as hot experts
        return self.apply_2_4_sparsity(weight)
    
    def sparsify_expert_weights(
        self,
        expert_weights: List[torch.Tensor],
        classification: Dict[str, List[int]]
    ) -> List[torch.Tensor]:
        """
        Apply adaptive sparsity to expert weights
        
        Args:
            expert_weights: List of expert weight tensors
            classification: Dict with 'hot' and 'cold' expert indices
            
        Returns:
            List of sparsified weight tensors
        """
        sparse_weights = []
        
        for expert_id, weight in enumerate(expert_weights):
            if expert_id in classification['hot']:
                # Hot expert: 2:4 sparsity
                sparse_weight = self.apply_2_4_sparsity(weight)
                print(f"[EXACT] Expert {expert_id} (HOT): Applied 2:4 sparsity")
            elif expert_id in classification['cold']:
                # Cold expert: 1:4 sparsity
                sparse_weight = self.apply_1_4_sparsity(weight)
                print(f"[EXACT] Expert {expert_id} (COLD): Applied 1:4 sparsity")
            else:
                # Should not happen if classification is complete
                sparse_weight = weight.clone()
                print(f"[EXACT] Expert {expert_id}: No sparsity applied (unclassified)")
            
            sparse_weights.append(sparse_weight)
        
        return sparse_weights
    
    def compute_sparsity_stats(
        self,
        weights: List[torch.Tensor],
        classification: Dict[str, List[int]]
    ) -> Dict:
        """
        Compute sparsity statistics
        
        Args:
            weights: List of sparsified weight tensors
            classification: Expert classification
            
        Returns:
            Dict with sparsity statistics
        """
        stats = {
            'num_experts': len(weights),
            'hot_experts': len(classification['hot']),
            'cold_experts': len(classification['cold']),
            'per_expert': {}
        }
        
        total_params = 0
        total_nonzero = 0
        
        for expert_id, weight in enumerate(weights):
            num_params = weight.numel()
            num_nonzero = weight.count_nonzero().item()
            sparsity = 1.0 - (num_nonzero / num_params)
            
            stats['per_expert'][expert_id] = {
                'params': num_params,
                'nonzero': num_nonzero,
                'sparsity': sparsity,
                'type': 'hot' if expert_id in classification['hot'] else 'cold'
            }
            
            total_params += num_params
            total_nonzero += num_nonzero
        
        stats['overall_sparsity'] = 1.0 - (total_nonzero / total_params)
        stats['total_params'] = total_params
        stats['total_nonzero'] = total_nonzero
        
        return stats
    
    def save_sparse_weights(
        self,
        sparse_weights: List[torch.Tensor],
        output_path: str,
        classification: Dict[str, List[int]]
    ):
        """
        Save sparsified weights to disk
        
        Args:
            sparse_weights: List of sparse weight tensors
            output_path: Path to save weights
            classification: Expert classification for metadata
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'weights': sparse_weights,
            'classification': classification,
            'config': {
                'hot_threshold': self.config.hot_threshold,
                'hot_sparsity': '2:4',
                'cold_sparsity': '1:4'
            }
        }
        
        torch.save(checkpoint, output_path)
        print(f"[EXACT] Saved sparse weights to {output_path}")
