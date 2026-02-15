#!/usr/bin/env python3
import os
import sys
import argparse
import gc
import torch

os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from deepseek.configuration_deepseek import DeepseekConfig
from deepseek.modeling_deepseek import DeepseekForCausalLM, DeepseekMoE
from deepseek.modeling_deepseek_samoyeds import SSDeepseekMoE

from exact_core import apply_exact_to_model, apply_uniform_samoyeds_sparsification
from exact_weight_loader_deepseek import load_pretrained_weights_into_samoyeds


def _convert_moe_to_samoyeds_inplace(model: DeepseekForCausalLM):
    """Replace `DeepseekMoE` blocks with `SSDeepseekMoE` and copy weights.

    - Gate weights are dense and copied directly.
    - Shared experts remain dense and are copied.
    - Routed experts' `gate_proj`, `up_proj`, `down_proj` weights are copied.
    """
    for layer in model.model.layers:
        if isinstance(layer.mlp, DeepseekMoE):
            ss_moe = SSDeepseekMoE(layer.mlp.config, skip_sparsifier=True)

            # Gate (routing network) stays dense
            ss_moe.gate.weight.data = layer.mlp.gate.weight.data.clone()

            # Shared experts (DeepSeek uses shared experts module)
            if hasattr(layer.mlp, 'shared_experts') and hasattr(ss_moe, 'shared_experts'):
                ss_moe.shared_experts.gate_proj.weight.data = layer.mlp.shared_experts.gate_proj.weight.data.clone()
                ss_moe.shared_experts.up_proj.weight.data = layer.mlp.shared_experts.up_proj.weight.data.clone()
                ss_moe.shared_experts.down_proj.weight.data = layer.mlp.shared_experts.down_proj.weight.data.clone()

            # Routed experts
            for i, expert in enumerate(layer.mlp.experts):
                ss_moe.experts[i].gate_proj.weight.data = expert.gate_proj.weight.data.clone()
                ss_moe.experts[i].up_proj.weight.data = expert.up_proj.weight.data.clone()
                ss_moe.experts[i].down_proj.weight.data = expert.down_proj.weight.data.clone()

            # Swap implementation
            layer.mlp = ss_moe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deepseek-ai/deepseek-moe-16b-base')
    parser.add_argument('--method', choices=['baseline', 'uniform_samoyeds', 'exact', 'test_pruning_only', 'test_hf_direct', 'test_custom_dense', 'test_random_weights', 'test_architecture_diff'], required=True,
                        help='baseline=dense, uniform_samoyeds=uniform 2:4, exact=adaptive compression')
    parser.add_argument('--profiling_json', default='artifacts/exact_profiling/deepseek-moe-16b-base_real_e64_layer.json',
                        help='Required for EXACT method')
    parser.add_argument('--use_modelopt', action='store_true', help='Use ModelOpt ASP for pruning')
    parser.add_argument('--tasks', default='mmlu,hellaswag,arc_challenge,winogrande')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("="*70)
    print(f"EVALUATION METHOD: {args.method.upper()}")
    print("="*70)

    if args.method == 'baseline':
        # Dense baseline - no compression
        print(f"Loading dense model from {args.model}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )
        model.eval()
        print("✓ Dense model loaded")

    elif args.method == 'uniform_samoyeds':
        # Uniform 2:4 baseline - all experts pruned equally
        print("Creating dense model for weight extraction...")
        config = DeepseekConfig.from_pretrained(args.model, trust_remote_code=True)
        dense_model = DeepseekForCausalLM(config).half()
        dense_model = load_pretrained_weights_into_samoyeds(dense_model, args.model)
        print("✓ Dense weights loaded")

        print("Creating Samoyeds model and converting MoE blocks...")
        samoyeds_model = DeepseekForCausalLM(config).half()
        _convert_moe_to_samoyeds_inplace(samoyeds_model)

        print(f"Applying uniform 2:4 sparsity (ModelOpt={args.use_modelopt})...")
        model = apply_uniform_samoyeds_sparsification(
            samoyeds_model,
            dense_model,
            use_modelopt=args.use_modelopt
        )

        # Free dense model
        del dense_model
        gc.collect()

        print("Moving to GPU...")
        model = model.cuda()
        model.eval()
        print("✓ Uniform Samoyeds model ready")

    elif args.method == 'exact':
        # EXACT adaptive compression
        if args.profiling_json is None:
            print("❌ profiling_json required for EXACT")
            sys.exit(1)

        print("Creating dense model for weight extraction...")
        config = DeepseekConfig.from_pretrained(args.model, trust_remote_code=True)
        dense_model = DeepseekForCausalLM(config).half()
        dense_model = load_pretrained_weights_into_samoyeds(dense_model, args.model)
        print("✓ Dense weights loaded")

        print("Creating Samoyeds model and converting MoE blocks...")
        samoyeds_model = DeepseekForCausalLM(config).half()
        _convert_moe_to_samoyeds_inplace(samoyeds_model)

        print(f"Applying EXACT adaptive compression from {args.profiling_json}...")
        print(f"  ModelOpt: {args.use_modelopt}")
        model = apply_exact_to_model(
            samoyeds_model,
            args.profiling_json,
            dense_model=dense_model,
            use_modelopt=args.use_modelopt
        )

        # Free uncompressed models
        del dense_model, samoyeds_model
        gc.collect()

        print("Moving to GPU...")
        model = model.cuda()
        model.eval()
        print("✓ EXACT model ready")

    elif args.method == 'test_pruning_only':
        # Test pruning WITHOUT Samoyeds - magnitude prune in-place
        print("Testing pruning accuracy without Samoyeds conversion...")
        config = DeepseekConfig.from_pretrained(args.model, trust_remote_code=True)
        model = DeepseekForCausalLM(config).half()  # Keep dense
        model = load_pretrained_weights_into_samoyeds(model, args.model)

        from exact_core.samoyed_converter import _prune_24_magnitude
        for layer in model.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                for expert in layer.mlp.experts:
                    expert.gate_proj.weight.data.copy_(_prune_24_magnitude(expert.gate_proj.weight))
                    expert.up_proj.weight.data.copy_(_prune_24_magnitude(expert.up_proj.weight))
                    expert.down_proj.weight.data.copy_(_prune_24_magnitude(expert.down_proj.weight))

        model = model.cuda()
        model.eval()
        print("✓ Pruned model ready (dense execution)")

    elif args.method == 'test_hf_direct':
        # Load HF model, prune in-place, NO custom weight loader
        print("Testing with direct HF loading...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )

        from exact_core.samoyed_converter import _prune_24_magnitude, _prune_14_magnitude
        for layer in model.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                for expert in layer.mlp.experts:
                    expert.gate_proj.weight.data.copy_(_prune_24_magnitude(expert.gate_proj.weight))
                    expert.up_proj.weight.data.copy_(_prune_24_magnitude(expert.up_proj.weight))
                    expert.down_proj.weight.data.copy_(_prune_24_magnitude(expert.down_proj.weight))

        model.eval()

    elif args.method == 'test_custom_dense':
        # Custom dense model with weight loader (no pruning)
        print("Testing custom dense model with weight loader (no pruning)...")
        config = DeepseekConfig.from_pretrained(args.model, trust_remote_code=True)
        model = DeepseekForCausalLM(config).half()
        model = load_pretrained_weights_into_samoyeds(model, args.model)
        print("✓ Custom dense model loaded")

        model = model.cuda()
        model.eval()
        print("✓ Custom dense model ready (no pruning)")

    elif args.method == 'test_random_weights':
        # Architecture sanity with random weights
        print("Testing DeepseekForCausalLM architecture with random weights...")
        config = DeepseekConfig.from_pretrained(args.model, trust_remote_code=True)
        model = DeepseekForCausalLM(config).half()
        print("✓ Random model created (no weight loading)")

        model = model.cuda()
        model.eval()
        print("✓ Random model ready")

    elif args.method == 'test_architecture_diff':
        print("Comparing HF vs Custom architecture outputs...")

        # Test input
        test_input = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        # Load HF official model (CPU, float32)
        print("Loading HF model...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            device_map='cpu',
            trust_remote_code=True
        ).eval()

        # Load custom model with same weights (CPU, float32)
        print("Loading custom model...")
        config = DeepseekConfig.from_pretrained(args.model, trust_remote_code=True)
        custom_model = DeepseekForCausalLM(config).float()
        custom_model = load_pretrained_weights_into_samoyeds(custom_model, args.model)
        custom_model.eval()

        # Debug: check training mode
        print(f"\nHF model training mode: {hf_model.training}")
        print(f"Custom model training mode: {custom_model.training}")
        print(f"HF layer 0 MoE training mode: {hf_model.model.layers[0].mlp.training}")
        print(f"Custom layer 0 MoE training mode: {custom_model.model.layers[0].mlp.training}")
        
        # Debug: compare MoE class types and implementations
        import inspect
        hf_moe = hf_model.model.layers[0].mlp
        custom_moe = custom_model.model.layers[0].mlp
        
        print(f"\nHF MoE class: {type(hf_moe)}")
        print(f"Custom MoE class: {type(custom_moe)}")
        print(f"HF MoE module: {type(hf_moe).__module__}")
        print(f"Custom MoE module: {type(custom_moe).__module__}")
        
        print("\n" + "="*70)
        print("HF MoE forward() source:")
        print("="*70)
        try:
            print(inspect.getsource(hf_moe.forward))
        except Exception as e:
            print(f"Could not get HF source: {e}")
        
        print("\n" + "="*70)
        print("Custom MoE forward() source:")
        print("="*70)
        try:
            print(inspect.getsource(custom_moe.forward))
        except Exception as e:
            print(f"Could not get Custom source: {e}")
        
        sys.exit(0)  # Exit after showing implementations
        if hasattr(hf_model.model.layers[0], 'mlp') and hasattr(hf_model.model.layers[0].mlp, 'experts'):
            print("\nWeight verification (Layer 0 MoE):")
            hf_moe = hf_model.model.layers[0].mlp
            custom_moe = custom_model.model.layers[0].mlp
            print(f"Gate max abs diff: {(hf_moe.gate.weight - custom_moe.gate.weight).abs().max():.8f}")
            print(f"Expert 0 gate_proj max abs diff: {(hf_moe.experts[0].gate_proj.weight - custom_moe.experts[0].gate_proj.weight).abs().max():.8f}")
            if hasattr(hf_moe, 'shared_experts') and hasattr(custom_moe, 'shared_experts'):
                print(f"Shared experts gate_proj max abs diff: {(hf_moe.shared_experts.gate_proj.weight - custom_moe.shared_experts.gate_proj.weight).abs().max():.8f}")

        # Record MoE outputs for first layer
        hf_moe_outs = []
        custom_moe_outs = []

        if hasattr(hf_model.model.layers[0], 'mlp'):
            original_hf_forward = hf_model.model.layers[0].mlp.forward
            def hf_wrapper(x):
                y = original_hf_forward(x)
                hf_moe_outs.append(y)
                return y
            hf_model.model.layers[0].mlp.forward = hf_wrapper

        if hasattr(custom_model.model.layers[0], 'mlp'):
            original_custom_forward = custom_model.model.layers[0].mlp.forward
            def custom_wrapper(x):
                y = original_custom_forward(x)
                custom_moe_outs.append(y)
                return y
            custom_model.model.layers[0].mlp.forward = custom_wrapper

        # Get HF output
        print("\nRunning HF model...")
        with torch.no_grad():
            hf_out = hf_model(test_input).logits
        print(f"HF output shape: {hf_out.shape}, mean: {hf_out.mean():.6f}")

        # Get custom output
        print("Running custom model...")
        with torch.no_grad():
            custom_out = custom_model(test_input).logits
        print(f"Custom output shape: {custom_out.shape}, mean: {custom_out.mean():.6f}")

        # Compare
        print("\n" + '='*70)
        print("ARCHITECTURE COMPARISON")
        print('='*70)
        print(f"Output diff (abs mean): {(hf_out - custom_out).abs().mean():.6f}")
        print(f"Output diff (max): {(hf_out - custom_out).abs().max():.6f}")
        if hf_moe_outs and custom_moe_outs:
            print(f"\nLayer 0 MoE output diff: {(hf_moe_outs[0] - custom_moe_outs[0]).abs().mean():.6f}")
            print(f"Layer 0 MoE output max diff: {(hf_moe_outs[0] - custom_moe_outs[0]).abs().max():.6f}")
        sys.exit(0)

    else:
        print(f"Unknown method: {args.method}")
        sys.exit(1)

    # Evaluate
    print(f"\nEvaluating on tasks: {args.tasks}")
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=args.tasks.split(','),
        batch_size=args.batch_size
    )
    print("✓ Evaluation complete")

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    def get_metric_value(task_result):
        # Extract primary accuracy metric
        if 'acc_norm,none' in task_result:
            return task_result['acc_norm,none']
        elif 'acc,none' in task_result:
            return task_result['acc,none']
        elif 'acc_norm' in task_result:
            return task_result['acc_norm']
        elif 'acc' in task_result:
            return task_result['acc']
        else:
            return next((v for k, v in task_result.items() if isinstance(v, (int, float)) and k != 'alias'), 0)

    print(f"\n{'Task':<20} {'Accuracy':<12}")
    print("-" * 35)
    for task in args.tasks.split(','):
        acc = get_metric_value(results['results'][task])
        print(f"{task:<20} {acc*100:>6.2f}%")


if __name__ == '__main__':
    main()
