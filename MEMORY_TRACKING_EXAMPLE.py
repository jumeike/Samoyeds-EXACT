"""
Example: How to use MemoryTracker in evaluate_exact_deepseek.py

Add these modifications to track memory during:
1. Model loading
2. Samoyeds conversion
3. EXACT compression
4. Inference
"""

# At the top of evaluate_exact_deepseek.py, add:
from memory_tracker import MemoryTracker, measure_model_memory, compare_model_sizes

# Initialize tracker
memory_tracker = MemoryTracker()

# ============================================================================
# TRACK BASELINE MODEL LOADING
# ============================================================================
if not args.skip_baseline:
    print("="*70)
    print("EVALUATING OFFICIAL PRETRAINED MODEL (BASELINE)")
    print("="*70)
    
    with memory_tracker.track('baseline_loading'):
        print(f"Loading {args.model}...")
        baseline_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )
        baseline_model.eval()
    
    # Measure baseline model size
    baseline_mem = measure_model_memory(baseline_model)
    print(f"\nBaseline Model Memory: {baseline_mem['total_gb']:.3f} GB")
    
    with memory_tracker.track('baseline_inference'):
        print("Evaluating baseline...")
        baseline_lm = HFLM(pretrained=baseline_model, tokenizer=tokenizer)
        baseline_results = evaluator.simple_evaluate(
            model=baseline_lm,
            tasks=args.tasks.split(','),
            batch_size=args.batch_size
        )
    
    print("✓ Baseline evaluation complete")
    
    # Free GPU memory
    print("Freeing GPU memory...")
    baseline_model = baseline_model.cpu()
    del baseline_model
    del baseline_lm
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("✓ GPU memory cleared")
else:
    baseline_results = None

# ============================================================================
# TRACK SAMOYEDS MODEL CREATION
# ============================================================================
print("\n" + "="*70)
print("EVALUATING SAMOYEDS DEEPSEEK WITH EXACT COMPRESSION")
print("="*70)

with memory_tracker.track('model_loading'):
    config = DeepseekConfig.from_pretrained(args.model, trust_remote_code=True)
    
    # Create dense model for baseline evaluation
    print("Creating Dense DeepSeek model...")
    dense_model = DeepseekForCausalLM(config).half()
    
    # Create Samoyeds DeepSeek model
    print("Creating Samoyeds DeepSeek model...")
    samoyeds_model = DeepseekForCausalLM(config).half()
    
    # Load pretrained weights into BOTH models
    print(f"Loading pretrained weights from {args.model}...")
    dense_model = load_pretrained_weights_into_samoyeds(dense_model, args.model)
    samoyeds_model = load_pretrained_weights_into_samoyeds(samoyeds_model, args.model)
    print("✓ Weights loaded")

# ============================================================================
# TRACK SAMOYEDS CONVERSION
# ============================================================================
with memory_tracker.track('conversion'):
    print("\nConverting MoE blocks to Samoyeds format...")
    from deepseek.modeling_deepseek import DeepseekMoE
    from deepseek.modeling_deepseek_samoyeds import SSDeepseekMoE
    
    for layer in samoyeds_model.model.layers:
        if isinstance(layer.mlp, DeepseekMoE):
            ss_moe = SSDeepseekMoE(layer.mlp.config)
            ss_moe.gate.weight.data = layer.mlp.gate.weight.data.clone()
            # ... rest of conversion code ...
            layer.mlp = ss_moe
    print("✓ Converted to Samoyeds format")

# ============================================================================
# TRACK EXACT COMPRESSION
# ============================================================================
with memory_tracker.track('exact_compression'):
    print(f"\nApplying EXACT compression from {args.profiling_json}...")
    exact_model = apply_exact_to_model(samoyeds_model, args.profiling_json, dense_model=dense_model)
    exact_model = exact_model.cuda()
    exact_model.eval()
    print("✓ EXACT compression applied")

# Compare model sizes
model_comparison = compare_model_sizes(dense_model, exact_model)

# ============================================================================
# TRACK INFERENCE
# ============================================================================
with memory_tracker.track('inference'):
    print("\nEvaluating EXACT compressed model...")
    exact_lm = HFLM(pretrained=exact_model, tokenizer=tokenizer)
    exact_results = evaluator.simple_evaluate(
        model=exact_lm,
        tasks=args.tasks.split(','),
        batch_size=args.batch_size
    )
    print("✓ EXACT evaluation complete")

# ============================================================================
# PRINT MEMORY SUMMARY
# ============================================================================
memory_tracker.print_summary()

# ============================================================================
# SAVE RESULTS WITH MEMORY METRICS
# ============================================================================
# ... existing results saving code ...

# Add memory metrics to CSV
if results_data:
    model_name = args.model.replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = 'artifacts/results/model_accuracy_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save accuracy results
    csv_filename = os.path.join(results_dir, f"exact_results_{model_name}_{timestamp}.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['task', 'baseline_acc', 'exact_acc', 'accuracy_drop']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_data)
    print(f"\nAccuracy results saved to: {csv_filename}")
    
    # Save memory metrics
    memory_csv = os.path.join(results_dir, f"memory_usage_{model_name}_{timestamp}.csv")
    memory_summary = memory_tracker.get_summary_dict()
    memory_summary.update(model_comparison)
    memory_summary['model'] = args.model
    memory_summary['tasks'] = args.tasks
    memory_summary['batch_size'] = args.batch_size
    
    with open(memory_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=memory_summary.keys())
        writer.writeheader()
        writer.writerow(memory_summary)
    print(f"Memory metrics saved to: {memory_csv}")
