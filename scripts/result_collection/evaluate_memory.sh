#!/bin/bash

gpu_id=${CUDA_VISIBLE_DEVICES:-0}
gpu_model=$(nvidia-smi -i $gpu_id --query-gpu=gpu_name --format=csv,noheader,nounits)
gpu_model=${gpu_model// /.}
cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)

echo "GPU model: $gpu_model"
echo "CUDA version: $cuda_version"

output_dir="artifacts/results/memory_csv"
mkdir -p $output_dir

# Configuration: model, batch_size, num_experts, intermediate_size, hidden_size, seq_len
# Mixtral 8x7B: 1, 8, 14336, 4096, 4096
# Mixtral 8x22B: 1, 8, 16384, 6144, 4096
# DeepSeek-V2: 16, 64, 1408, 2048, 4096
# Qwen2-MoE: 14, 60, 1408, 2048, 4096
config="mixtral,1,8,14336,4096,4096 \
mixtral,1,8,16384,6144,4096 \
deepseek,16,64,1408,2048,4096 \
qwen2_moe,14,60,1408,2048,4096"

OLD_IFS="$IFS"

count=0
total=$(echo -e "$config" | awk -v RS=' ' 'END {print NR}')

start_string="model,batch_size,num_experts,intermediate_size,hidden_size,seq_len,exact_memory_mb,dense_memory_mb,samoyeds_memory_mb,exact_vs_dense_savings_pct,exact_vs_samoyeds_savings_pct"

output_file="$output_dir/memory_evaluation.csv"
echo $start_string > "${output_file}"

echo ""
echo "========================================="
echo "Memory Evaluation Configuration"
echo "========================================="
echo "Models: $total"
echo "Implementations: EXACT vs Dense (transformers) vs Samoyeds"
echo "========================================="

IFS=" "
for cfg in $config; do
    IFS=","; set -- $cfg
    model=$1; batch=$2; expert=$3; intermediate_size=$4; hidden_size=$5; seq_len=$6

    ((count++))
    echo ""
    echo "========================================="
    echo "Progress: ${count}/${total}"
    echo "Evaluating ${model}"
    echo "Config: batch=${batch}, experts=${expert}, intermediate=${intermediate_size}, hidden=${hidden_size}, seq_len=${seq_len}"
    echo "========================================="

    # Run EXACT and capture memory
    echo "  Measuring EXACT memory usage..."
    exact_mem=$(python scripts/result_collection/measure_memory_impl.py \
        --model ${model} \
        --implementation exact \
        --batch_size ${batch} \
        --num_experts ${expert} \
        --intermediate_size ${intermediate_size} \
        --hidden_size ${hidden_size} \
        --seq_len ${seq_len} 2>/dev/null || echo "0.00")

    # Run Dense (transformers) and capture memory
    echo "  Measuring Dense (transformers) memory usage..."
    dense_mem=$(python scripts/result_collection/measure_memory_impl.py \
        --model ${model} \
        --implementation dense \
        --batch_size ${batch} \
        --num_experts ${expert} \
        --intermediate_size ${intermediate_size} \
        --hidden_size ${hidden_size} \
        --seq_len ${seq_len} 2>/dev/null || echo "0.00")

    # Run Samoyeds and capture memory
    echo "  Measuring Samoyeds memory usage..."
    samoyeds_mem=$(python scripts/result_collection/measure_memory_impl.py \
        --model ${model} \
        --implementation samoyeds \
        --batch_size ${batch} \
        --num_experts ${expert} \
        --intermediate_size ${intermediate_size} \
        --hidden_size ${hidden_size} \
        --seq_len ${seq_len} 2>/dev/null || echo "0.00")

    # Calculate savings percentages
    if (( $(echo "$dense_mem > 0" | bc -l) )); then
        exact_vs_dense_savings=$(echo "scale=2; (($dense_mem - $exact_mem) / $dense_mem) * 100" | bc)
    else
        exact_vs_dense_savings="0.00"
    fi

    if (( $(echo "$samoyeds_mem > 0" | bc -l) )); then
        exact_vs_samoyeds_savings=$(echo "scale=2; (($samoyeds_mem - $exact_mem) / $samoyeds_mem) * 100" | bc)
    else
        exact_vs_samoyeds_savings="0.00"
    fi

    echo "  Results:"
    echo "    EXACT:    ${exact_mem} MB"
    echo "    Dense:    ${dense_mem} MB"
    echo "    Samoyeds: ${samoyeds_mem} MB"
    echo "    Savings vs Dense:    ${exact_vs_dense_savings}%"
    echo "    Savings vs Samoyeds: ${exact_vs_samoyeds_savings}%"

    # Write to CSV
    echo "${model},${batch},${expert},${intermediate_size},${hidden_size},${seq_len},${exact_mem},${dense_mem},${samoyeds_mem},${exact_vs_dense_savings},${exact_vs_samoyeds_savings}" >> $output_file

    echo "  Completed ${model} memory evaluation"
done

IFS="$OLD_IFS"

echo ""
echo "========================================="
echo "Memory evaluation complete!"
echo "Results saved to: ${output_file}"
echo "========================================="
