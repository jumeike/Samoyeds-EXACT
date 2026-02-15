#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-mmlu_astronomy}"
# STEPS="${2:-2000}"
STEPS_07="${2:-2000}"
STEPS_08="${3:-2000}"
STEPS_09="${4:-2000}"
STEPS_UNI="${5:-2000}"

MODEL="Qwen/Qwen1.5-MoE-A2.7B"
BASE_JSON_DIR="artifacts/exact_profiling/qwen1.5-moe-a2.7b"
# MODEL="deepseek-ai/deepseek-moe-16b-base"
# BASE_JSON_DIR="artifacts/exact_profiling/deepseek-moe-16b-base"
OUT_DIR="artifacts/checkpoints"

run_exact () {
  local MASS="$1"
  local STEPS="$2"
  echo "Running: ${MODEL} | task=${TASK} | steps=${STEPS} | mass=${MASS} | mode=exact"
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune_exact_masks_fast.py \
    --model "$MODEL" \
    --profiling_json "${BASE_JSON_DIR}/deepseek-moe-16b-base_eval_tasks_${TASK}_mass${MASS}_real_e64_layer.json" \
    --fsdp --fsdp_use_orig_params --fsdp_min_num_params 10000000 \
    --grad_ckpt --optimizer adafactor \
    --train_mode experts \
    --dataset eval_tasks --eval_tasks "$TASK" \
    --steps "${STEPS}" --batch_size 1 --grad_accum 4 --max_length 512 --lr 5e-6 \
    --amp_dtype bf16 \
    --save_mode sharded \
    --save_path "${OUT_DIR}/finetuned_exact_deepseek_${TASK}_${STEPS}steps_experts_train_mode_mass${MASS}_fast"
}

run_uniform () {
  local STEPS="$1"
  echo "Running: ${MODEL} | task=${TASK} | steps=${STEPS} | mode=uniform (cold_as_24)"
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 finetune_exact_masks_fast.py \
    --model "$MODEL" \
    --profiling_json "${BASE_JSON_DIR}/deepseek-moe-16b-base_eval_tasks_${TASK}_mass0.9_real_e64_layer.json" \
    --fsdp --fsdp_use_orig_params --fsdp_min_num_params 10000000 \
    --grad_ckpt --optimizer adafactor \
    --train_mode experts \
    --dataset eval_tasks --eval_tasks "$TASK" \
    --steps "${STEPS}" --batch_size 1 --grad_accum 4 --max_length 512 --lr 5e-6 \
    --amp_dtype bf16 \
    --save_mode sharded --cold_as_24 \
    --save_path "${OUT_DIR}/finetuned_uniform_deepseek_${TASK}_${STEPS}steps_experts_train_mode_mass0.9_fast"
}

# run_exact 0.7 "${STEPS_07}"
# run_exact 0.8 "${STEPS_08}"
run_exact 0.7 "${STEPS_07}"
# run_uniform "${STEPS_UNI}"
# run_exact 0.9 "${STEPS_09}"
