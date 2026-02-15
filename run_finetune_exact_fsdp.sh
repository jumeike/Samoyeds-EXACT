#!/usr/bin/env bash
set -euo pipefail

# Example 4-GPU FSDP finetune for EXACT masks.
# Edit the command below as needed.

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune_exact_masks.py \
  --model Qwen/Qwen1.5-MoE-A2.7B \
  --profiling_json artifacts/exact_profiling/qwen1.5-moe-a2.7b_eval_tasks_mmlu_astronomy_mass0.8_real_e60_layer.json \
  --fsdp \
  --fsdp_use_orig_params \
  --fsdp_min_num_params 10000000 \
  --grad_ckpt \
  --optimizer adafactor \
  --train_mode experts \
  --dataset eval_tasks \
  --eval_tasks_eval mmlu_astronomy \
  --eval_batch_size 1 \
  --steps 100 \
  --batch_size 1 \
  --grad_accum 4 \
  --max_length 512 \
  --lr 5e-6 \
  --save_mode full \
  --save_path artifacts/finetuned_exact_mmlu_100_experts_train_mode_mass0.8
    # --eval_tasks mmlu_astronomy \
  # --eval_every 1000 \
