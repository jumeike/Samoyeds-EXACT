#!/bin/bash

gpu_id=${CUDA_VISIBLE_DEVICES:-0}
gpu_model=$(nvidia-smi -i $gpu_id --query-gpu=gpu_name --format=csv,noheader,nounits)
gpu_model=${gpu_model// /.}
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)

echo "GPU model: $gpu_model"
echo "CUDA version: $cuda_version"

output_dir="artifacts/results/throughput_csv"
mkdir -p $output_dir

# Configuration: model, num_experts, intermediate_size, hidden_size, seq_len
# Batch sizes will be tested: 1, 2, 4, 8, 16, 32, 64 (powers of 2)
# Mixtral 8x7B: 8, 14336, 4096, 4096
# Mixtral 8x22B: 8, 16384, 6144, 4096
# DeepSeek-V2: 64, 1408, 2048, 4096
# Qwen2-MoE: 60, 1408, 2048, 4096
config="mixtral,8,14336,4096,4096 \
mixtral,8,16384,6144,4096 \
deepseek,64,1408,2048,4096 \
qwen2_moe,60,5632,2048,4096"

# Batch sizes to test (powers of 2)
batch_sizes="1 2 4 8 16 32 64"

OLD_IFS="$IFS"

# Calculate total iterations for progress tracking
total_iterations=0
num_configs=$(echo -e "$config" | awk -v RS=' ' 'END {print NR}')
num_batch_sizes=$(echo $batch_sizes | wc -w)
total_iterations=$((num_configs * num_batch_sizes))

count=0

start_string="model,model_type,kernel_type,iter,batch_size,seq_len,hidden_size,intermediate_size,expert_num,time,atten_mode"

output_file="$output_dir/throughput_evaluation.csv"
echo $start_string > "${output_file}"

echo ""
echo "========================================="
echo "Throughput Evaluation Configuration"
echo "========================================="
echo "Models: $num_configs"
echo "Batch sizes: $batch_sizes"
echo "Total evaluations: $total_iterations × 4 implementations = $((total_iterations * 4))"
echo "========================================="

IFS=" "
for cfg in $config; do
    IFS=","; set -- $cfg
    model=$1; expert=$2; intermediate_size=$3; hidden_size=$4; seq_len=$5

    echo ""
    echo "========================================="
    echo "Evaluating ${model}"
    echo "Config: experts=${expert}, intermediate_size=${intermediate_size}, hidden_size=${hidden_size}, seq_len=${seq_len}"
    echo "========================================="

    IFS=" "
    for batch in $batch_sizes; do
        ((count++))
        echo ""
        echo "Progress: ${count}/${total_iterations} (${model} batch_size=${batch})"

        echo "  Running EXACT implementation..."
        python ${model}_EXACT.py --time --batch_size $batch --layer --flash --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len >> $output_file

        echo "  Running Samoyeds implementation..."
        python ${model}_Samoyeds.py --time --batch_size $batch --layer --flash --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len >> $output_file

        echo "  Running megablocks implementation..."
        python ${model}_megablocks.py --time --batch_size $batch --layer --flash --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len >> $output_file

        echo "  Running transformers implementation..."
        python ${model}_transformers.py --time --batch_size $batch --layer --flash --experts $expert --hidden_size $hidden_size --intermediate_size $intermediate_size --seq_len $seq_len >> $output_file
    done
    
    echo ""
    echo "Completed ${model} evaluation"
done

IFS="$OLD_IFS"

echo ""
echo "========================================="
echo "Adding tokens_per_sec column to CSV..."
echo "========================================="
python scripts/result_collection/add_throughput_column.py "${output_file}"

echo ""
echo "========================================="
echo "Throughput evaluation complete!"
echo "Results saved to: ${output_file}"
echo "========================================="
