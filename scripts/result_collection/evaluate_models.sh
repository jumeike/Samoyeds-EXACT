#!/bin/bash

gpu_id=${CUDA_VISIBLE_DEVICES:-0}
gpu_model=$(nvidia-smi -i $gpu_id --query-gpu=gpu_name --format=csv,noheader,nounits)
gpu_model=${gpu_model// /.}
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)

echo "GPU model: $gpu_model"
echo "CUDA version: $cuda_version"

output_dir="artifacts/results/model_evaluation_csv"
mkdir -p $output_dir

# Configuration: model, batch_size, num_experts, intermediate_size, hidden_size
# Mixtral 8x7B: 1, 8, 14336, 4096
# Mixtral 8x22B: 1, 8, 16384, 6144
# DeepSeek-V2: 16, 64, 1408, 2048
# Qwen2-MoE: 14, 60, 1408, 2048
config="mixtral,1,8,14336,4096 \
mixtral,1,8,16384,6144 \
deepseek,16,64,1408,2048 \
qwen2_moe,14,60,5632,2048"

OLD_IFS="$IFS"

count=0
total=$(echo -e "$config" | awk -v RS=' ' 'END {print NR}')

start_string="model,model_type,kernel_type,iter,batch_size,seq_len,hidden_size,intermediate_size,expert_num,time,atten_mode"

output_file="$output_dir/model_evaluation.csv"
echo $start_string > "${output_file}"

IFS=" "
for cfg in $config; do
    IFS=","; set -- $cfg
    model=$1; batch=$2; expert=$3; intermediate_size=$4; hidden_size=$5;

    ((count++))
    echo "Progress: ${count}/${total} (Running ${model} with batch_size=${batch} num_experts=${expert} intermediate_size=${intermediate_size} hidden_size=${hidden_size})"

    echo "Running EXACT implementation..."
    python ${model}_EXACT.py --time --batch_size $batch --layer --flash --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size >> $output_file

    echo "Running Samoyeds implementation..."
    python ${model}_Samoyeds.py --time --batch_size $batch --layer --flash --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size >> $output_file

    echo "Running megablocks implementation..."
    python ${model}_megablocks.py --time --batch_size $batch --layer --flash --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size >> $output_file

    echo "Running transformers implementation..."
    python ${model}_transformers.py --time --batch_size $batch --layer --flash --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size >> $output_file

    echo "Completed ${model} evaluation"
    echo "---"
done

IFS="$OLD_IFS"

echo ""
echo "Evaluation complete! Results saved to: ${output_file}"
echo "Total configurations evaluated: ${total}"

