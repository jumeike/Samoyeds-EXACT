#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-mmlu_astronomy}"
STEPS="${2:-2000}"

MODEL="mistralai/Mixtral-8x7B-v0.1"
BASE_JSON_DIR="artifacts/exact_profiling/mixtral-8x7b-v0.1"
OUT_DIR="artifacts/checkpoints"

run_exact () {
  local MASS="$1"
  echo "Running: ${MODEL} | task=${TASK} | steps=${STEPS} | mass=${MASS} | mode=exact"
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune_exact_masks_fast.py \
    --model "$MODEL" \
    --profiling_json "${BASE_JSON_DIR}/mixtral-8x7b-v0.1_eval_tasks_${TASK}_mass${MASS}_real_e8_layer.json" \
    --fsdp --fsdp_use_orig_params --fsdp_min_num_params 10000000 \
    --grad_ckpt --optimizer adafactor \
    --train_mode experts \
    --dataset eval_tasks --eval_tasks "$TASK" \
    --steps "${STEPS}" --batch_size 1 --grad_accum 4 --max_length 512 --lr 5e-6 \
    --save_mode full \
    --save_path "${OUT_DIR}/finetuned_exact_mixtral_${TASK}_${STEPS}steps_experts_train_mode_mass${MASS}_fast"
}

run_uniform () {
  echo "Running: ${MODEL} | task=${TASK} | steps=${STEPS} | mode=uniform (cold_as_24)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune_exact_masks_fast.py \
    --model "$MODEL" \
    --profiling_json "${BASE_JSON_DIR}/mixtral-8x7b-v0.1_eval_tasks_${TASK}_mass0.9_real_e8_layer.json" \
    --fsdp --fsdp_use_orig_params --fsdp_min_num_params 10000000 \
    --grad_ckpt --optimizer adafactor \
    --train_mode experts \
    --dataset eval_tasks --eval_tasks "$TASK" \
    --steps "${STEPS}" --batch_size 1 --grad_accum 4 --max_length 512 --lr 5e-6 \
    --save_mode full --cold_as_24 \
    --save_path "${OUT_DIR}/finetuned_uniform_mixtral_${TASK}_${STEPS}steps_experts_train_mode_mass0.9_fast"
}

run_exact 0.7
# run_exact 0.8
# run_exact 0.9
# run_uniform
