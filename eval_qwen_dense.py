#!/usr/bin/env python3
"""
HF-only evaluation for pruning accuracy (no Samoyeds kernels).
Methods:
  - baseline: dense HF model
  - uniform_dense: uniform 2:4 pruning on experts
  - exact_dense: adaptive hot/cold pruning from profiling JSON
"""
import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import argparse
import json
import torch
import torch.nn.functional as F
import types
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset

from exact_core.samoyed_converter import _prune_24_magnitude, _prune_14_magnitude, _prune_with_modelopt_24
from exact_calibration import collect_calibration_activations, prune_expert_weights, get_moe_layers


def apply_uniform_dense(model):
    """Apply uniform 2:4 pruning to experts in-place (HF dense compute)."""
    print("Applying uniform 2:4 pruning (dense compute)...")
    for layer in model.model.layers:
        if not hasattr(layer.mlp, 'experts'):
            continue
        for expert in layer.mlp.experts:
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                weight = getattr(expert, name).weight
                pruned = _prune_24_magnitude(weight.data)
                getattr(expert, name).weight.data.copy_(pruned.to(weight.device, dtype=weight.dtype))
    return model


def apply_exact_dense(
    model,
    profiling_json,
    cold_as_24=False,
    calib_acts=None,
    sgpt_inputs=None,
    use_modelopt_mag=False,
    use_modelopt_sgpt=False,
):
    """Apply EXACT hot/cold pruning to experts in-place (HF dense compute)."""
    print(f"Applying EXACT pruning (dense compute) from {profiling_json}...")
    with open(profiling_json) as f:
        prof = json.load(f)
    if 'layers' not in prof:
        raise ValueError(f"Expected 'layers' key. Got: {list(prof.keys())}")
    layers_data = prof['layers']

    for l_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, 'experts'):
            continue
        layer_key = str(l_idx)
        if layer_key not in layers_data:
            continue
        classification = layers_data[layer_key].get('classification', {})
        hot_experts = set(classification.get('hot', []))
        cold_experts = set(classification.get('cold', []))

        for e_idx, expert in enumerate(layer.mlp.experts):
            if calib_acts is not None:
                act = calib_acts.get((l_idx, e_idx))
                if act is None:
                    act = torch.ones(expert.gate_proj.weight.shape[1])
                if e_idx in hot_experts:
                    prune_expert_weights(expert, act, n=2, m=4)
                elif e_idx in cold_experts:
                    prune_expert_weights(expert, act, n=1, m=4)
                else:
                    prune_expert_weights(expert, act, n=2, m=4)
                continue

            for name in ['gate_proj', 'up_proj', 'down_proj']:
                weight = getattr(expert, name).weight
                if e_idx in hot_experts:
                    if use_modelopt_sgpt:
                        calib = None
                        if sgpt_inputs is not None:
                            calib = sgpt_inputs.get((l_idx, e_idx, name))
                        if calib is None:
                            print(f"WARNING: Missing SGPT calib for layer {l_idx} expert {e_idx} {name}; using magnitude.")
                            pruned = _prune_24_magnitude(weight.data)
                        else:
                            pruned = _prune_with_modelopt_24(weight.data, calib_data=calib)
                    elif use_modelopt_mag:
                        pruned = _prune_with_modelopt_24(weight.data)
                    else:
                        pruned = _prune_24_magnitude(weight.data)
                elif e_idx in cold_experts:
                    if cold_as_24:
                        if use_modelopt_sgpt:
                            calib = None
                            if sgpt_inputs is not None:
                                calib = sgpt_inputs.get((l_idx, e_idx, name))
                            if calib is None:
                                print(f"WARNING: Missing SGPT calib for layer {l_idx} expert {e_idx} {name}; using magnitude.")
                                pruned = _prune_24_magnitude(weight.data)
                            else:
                                pruned = _prune_with_modelopt_24(weight.data, calib_data=calib)
                        elif use_modelopt_mag:
                            pruned = _prune_with_modelopt_24(weight.data)
                        else:
                            pruned = _prune_24_magnitude(weight.data)
                    else:
                        pruned = _prune_14_magnitude(weight.data)
                else:
                    pruned = _prune_24_magnitude(weight.data)
                weight.data.copy_(pruned.to(weight.device, dtype=weight.dtype))
    return model


def _expert_forward_with_masks(expert, x, masks, device):
    w_gate = expert.gate_proj.weight
    w_up = expert.up_proj.weight
    w_down = expert.down_proj.weight
    m_gate = masks["gate_proj"].to(device=device, dtype=w_gate.dtype, non_blocking=True)
    m_up = masks["up_proj"].to(device=device, dtype=w_up.dtype, non_blocking=True)
    m_down = masks["down_proj"].to(device=device, dtype=w_down.dtype, non_blocking=True)
    return F.linear(
        expert.act_fn(F.linear(x, w_gate * m_gate)) * F.linear(x, w_up * m_up),
        w_down * m_down,
    )


def _build_radm_masks(model, cold_as_24=False):
    """Precompute hot/cold masks per expert for RADM on CPU to reduce GPU memory."""
    radm_masks = {}
    for l_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, 'experts'):
            continue
        layer_masks = {}
        for e_idx, expert in enumerate(layer.mlp.experts):
            expert_masks = {"hot": {}, "cold": {}}
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                weight = getattr(expert, name).weight.data.detach().cpu()
                hot_w = _prune_24_magnitude(weight, return_to_device=False)
                cold_w = _prune_24_magnitude(weight, return_to_device=False) if cold_as_24 else _prune_14_magnitude(weight, return_to_device=False)
                expert_masks["hot"][name] = (hot_w != 0)
                expert_masks["cold"][name] = (cold_w != 0)
            layer_masks[e_idx] = expert_masks
        radm_masks[l_idx] = layer_masks
    return radm_masks


def apply_radm_dense(model, radm_json, cold_as_24=False):
    """Apply RADM (token-level hot/cold) by patching MoE forward."""
    print(f"Applying RADM (dense compute) from {radm_json}...")
    with open(radm_json) as f:
        prof = json.load(f)
    tau_per_layer = {}
    if 'radm' in prof and 'tau_per_layer' in prof['radm']:
        tau_per_layer = {int(k): float(v) for k, v in prof['radm']['tau_per_layer'].items()}
    elif 'layers' in prof:
        for k, v in prof['layers'].items():
            if isinstance(v, dict) and 'radm' in v and 'tau' in v['radm']:
                tau_per_layer[int(k)] = float(v['radm']['tau'])

    radm_masks = _build_radm_masks(model, cold_as_24=cold_as_24)

    for l_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, 'experts'):
            continue
        moe = layer.mlp
        if not hasattr(moe, "gate"):
            continue
        setattr(moe, "_radm_layer_idx", l_idx)

        def _radm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            router_logits = self.gate(hidden_states)

            routing_probs = F.softmax(router_logits, dim=1, dtype=torch.float)
            topk_vals, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
            topk_vals_raw = topk_vals
            if self.norm_topk_prob:
                topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
            routing_weights = topk_vals.to(hidden_states.dtype)

            # Token-level confidence margin
            if topk_vals_raw.shape[1] >= 2:
                margin = (topk_vals_raw[:, 0] - topk_vals_raw[:, 1]).to(hidden_states.dtype)
            else:
                margin = topk_vals_raw[:, 0].to(hidden_states.dtype)
            tau = tau_per_layer.get(getattr(self, "_radm_layer_idx", 0), 0.0)
            hot_token = margin >= tau

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            layer_idx = getattr(self, "_radm_layer_idx", 0)
            for expert_idx in range(self.num_experts):
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])

                if top_x.shape[0] == 0:
                    continue

                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                cur_hot = hot_token[top_x]

                if cur_hot.any():
                    hot_idx = cur_hot.nonzero(as_tuple=False).squeeze(-1)
                    x_hot = current_state[hot_idx]
                    rw_hot = routing_weights[top_x[hot_idx], idx[hot_idx], None]
                    masks_hot = radm_masks[layer_idx][expert_idx]["hot"]
                    out_hot = _expert_forward_with_masks(expert_layer, x_hot, masks_hot, hidden_states.device) * rw_hot
                    final_hidden_states.index_add_(0, top_x[hot_idx], out_hot.to(hidden_states.dtype))

                if (~cur_hot).any():
                    cold_idx = (~cur_hot).nonzero(as_tuple=False).squeeze(-1)
                    x_cold = current_state[cold_idx]
                    rw_cold = routing_weights[top_x[cold_idx], idx[cold_idx], None]
                    masks_cold = radm_masks[layer_idx][expert_idx]["cold"]
                    out_cold = _expert_forward_with_masks(expert_layer, x_cold, masks_cold, hidden_states.device) * rw_cold
                    final_hidden_states.index_add_(0, top_x[cold_idx], out_cold.to(hidden_states.dtype))

            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            final_hidden_states = final_hidden_states + shared_expert_output

            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

        moe.forward = types.MethodType(_radm_forward, moe)

    return model


def get_acc(r):
    for k in ['acc_norm,none', 'acc,none', 'acc_norm', 'acc']:
        if k in r:
            return r[k]
    return 0


def collect_sgpt_inputs(model, tokenizer, calib_texts, moe_layers, max_tokens=1024, max_length=512):
    """Collect per-expert per-projection inputs for SparseGPT."""
    hooks = []
    collected = {}
    counts = {}

    def _append(key, x):
        if x is None:
            return
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim == 2:
            pass
        else:
            return
        x = x.detach().float().cpu()
        cur = counts.get(key, 0)
        if cur >= max_tokens:
            return
        if cur + x.shape[0] > max_tokens:
            x = x[: max_tokens - cur]
        if key not in collected:
            collected[key] = x
        else:
            collected[key] = torch.cat([collected[key], x], dim=0)
        counts[key] = collected[key].shape[0]

    for layer_idx, moe in enumerate(moe_layers):
        for expert_idx, expert in enumerate(moe.experts):
            for name in ['gate_proj', 'up_proj', 'down_proj']:
                if not hasattr(expert, name):
                    continue
                linear = getattr(expert, name)

                def _make_hook(l_idx, e_idx, proj):
                    def hook(module, input, output):
                        key = (l_idx, e_idx, proj)
                        if key in counts and counts[key] >= max_tokens:
                            return
                        x = input[0] if input else None
                        _append(key, x)
                    return hook

                hooks.append(linear.register_forward_hook(_make_hook(layer_idx, expert_idx, name)))

    model.eval()
    print(f"Collecting SGPT inputs on {len(calib_texts)} samples...")
    with torch.no_grad():
        for text in calib_texts:
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)

    for h in hooks:
        h.remove()

    return collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen1.5-MoE-A2.7B')
    parser.add_argument('--method', choices=['baseline', 'uniform_dense', 'exact_dense', 'radm_dense'], required=True)
    parser.add_argument('--profiling_json', default='artifacts/exact_profiling/qwen1.5-moe-a2.7b_real_e60_layer.json')
    parser.add_argument('--radm_json', default=None, help='RADM profiling JSON (tau per layer)')
    parser.add_argument('--tasks', default='mmlu_astronomy')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cold_as_24', action='store_true', help='Force cold experts to 2:4 pruning')
    parser.add_argument('--skip_prune', action='store_true',
                        help='Skip applying pruning (use if checkpoint already has masks applied)')
    parser.add_argument('--use_calibration', action='store_true', help='Use calibration (SparseGPT-like) pruning')
    parser.add_argument('--use_modelopt_mag', action='store_true', help='Use ModelOpt magnitude pruning for 2:4')
    parser.add_argument('--use_modelopt_sgpt', action='store_true', help='Use ModelOpt SparseGPT for 2:4 (uses calib)')
    parser.add_argument('--calib_dataset', default='c4', choices=['wikitext', 'c4', 'eval_tasks'])
    parser.add_argument('--calib_samples', type=int, default=512)
    parser.add_argument('--calib_max_length', type=int, default=512)
    parser.add_argument('--calib_eval_tasks', default='arc_challenge,winogrande,mmlu_astronomy,openbookqa,gsm8k')
    parser.add_argument('--sgpt_max_tokens', type=int, default=1024,
                        help='Max tokens per expert/proj for SGPT calibration inputs')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("=" * 70)
    print(f"METHOD: {args.method.upper()}")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    print("✓ Loaded")

    calib_acts = None
    sgpt_inputs = None
    if (args.use_calibration or args.use_modelopt_sgpt) and args.method == 'exact_dense':
        print(f"Collecting calibration activations from {args.calib_dataset} ({args.calib_samples} samples)...")
        calib_texts = []
        if args.calib_dataset == 'wikitext':
            ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            for row in ds:
                if row.get('text', '').strip():
                    calib_texts.append(row['text'])
                if len(calib_texts) >= args.calib_samples:
                    break
        elif args.calib_dataset == 'c4':
            ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
            for row in ds:
                if row.get('text', '').strip():
                    calib_texts.append(row['text'])
                if len(calib_texts) >= args.calib_samples:
                    break
        else:
            tasks = [t.strip() for t in args.calib_eval_tasks.split(',') if t.strip()]
            for t in tasks:
                if t == 'arc_challenge':
                    ds = load_dataset('ai2_arc', 'ARC-Challenge', split='test')
                    for row in ds:
                        q = row.get('question', '')
                        choices = row.get('choices', {})
                        if isinstance(choices, dict) and 'text' in choices and 'label' in choices:
                            c = "\n".join([f"{l}. {tx}" for l, tx in zip(choices['label'], choices['text'])])
                        else:
                            c = ""
                        calib_texts.append(f"{q}\n{c}")
                        if len(calib_texts) >= args.calib_samples:
                            break
                if len(calib_texts) >= args.calib_samples:
                    break
        moe_layers = get_moe_layers(model)
        if args.use_modelopt_sgpt:
            # Collect per-projection inputs for SGPT
            sgpt_inputs = collect_sgpt_inputs(
                model,
                tokenizer,
                calib_texts,
                moe_layers,
                max_tokens=args.sgpt_max_tokens,
                max_length=args.calib_max_length,
            )
        else:
            calib_acts = collect_calibration_activations(model, tokenizer, calib_texts, moe_layers)

    if args.method == 'uniform_dense':
        if not args.skip_prune:
            model = apply_uniform_dense(model)
    elif args.method == 'exact_dense':
        if not args.skip_prune:
            model = apply_exact_dense(
                model,
                args.profiling_json,
                cold_as_24=args.cold_as_24,
                calib_acts=calib_acts if (args.use_calibration or args.use_modelopt_sgpt) else None,
                sgpt_inputs=sgpt_inputs,
                use_modelopt_mag=args.use_modelopt_mag,
                use_modelopt_sgpt=args.use_modelopt_sgpt,
            )
    elif args.method == 'radm_dense':
        if not args.skip_prune:
            radm_json = args.radm_json or args.profiling_json
            model = apply_radm_dense(
                model,
                radm_json,
                cold_as_24=args.cold_as_24,
            )

    model.eval()

    print(f"\nEvaluating: {args.tasks}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(model=lm, tasks=args.tasks.split(','), batch_size=args.batch_size)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n{'Task':<20} {'Accuracy'}")
    print("-" * 35)
    for task in args.tasks.split(','):
        print(f"{task:<20} {get_acc(results['results'][task]) * 100:>6.2f}%")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"MAX_MEMORY_ALLOCATED_GB: {peak_gb:.3f}")


if __name__ == "__main__":
    main()
