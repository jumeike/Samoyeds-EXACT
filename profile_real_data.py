import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from exact_core.expert_tracker import ExpertUsageTracker
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='mistralai/Mixtral-8x7B-v0.1, Qwen/Qwen1.5-MoE-A2.7B, or deepseek-ai/deepseek-moe-16b-base')
parser.add_argument('--dataset', default='wikitext', choices=['wikitext', 'c4', 'pile', 'eval_tasks'])
parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
parser.add_argument('--max_length', type=int, default=512)
parser.add_argument('--hot_threshold', type=float, default=0.02,
                    help='Absolute activation_rate threshold (default mode)')
parser.add_argument('--classify_mode', choices=['threshold', 'relative', 'mass', 'percentile'], default='threshold',
                    help='Expert classification mode')
parser.add_argument('--relative_delta', type=float, default=1.0,
                    help='Relative threshold over uniform rate: hot if >= (1/N)*(1+delta)')
parser.add_argument('--mass', type=float, default=0.6,
                    help='Cumulative routing mass to mark hot (mass-based mode)')
parser.add_argument('--percentile', type=float, default=90.0,
                    help='Percentile for thresholding activation_rate (percentile mode)')
parser.add_argument('--min_hot', type=int, default=1,
                    help='Minimum hot experts per layer (relative/mass modes)')
parser.add_argument('--eval_tasks', default='arc_challenge,winogrande,mmlu_astronomy,openbookqa,gsm8k',
                    help='Comma-separated eval tasks for task-specific profiling')
parser.add_argument('--per_task', action='store_true',
                    help='When using eval_tasks, save a separate profile per task')
parser.add_argument('--routing_source', choices=['auto', 'indices', 'logits'], default='auto',
                    help='How to extract routing: auto prefers indices if present, else logits')
parser.add_argument('--routing_debug', action='store_true',
                    help='Print a one-time routing source debug message')
parser.add_argument('--radm', action='store_true',
                    help='Also collect router confidence margins for RADM and save tau per layer')
parser.add_argument('--radm_hot_ratio', type=float, default=0.5,
                    help='Target hot ratio for RADM (fraction of tokens treated as hot)')
parser.add_argument('--radm_max_margins', type=int, default=200000,
                    help='Max margin samples to keep per layer for RADM (reservoir)')
args = parser.parse_args()

# Load model
print(f"Loading {args.model}...")
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True 
)
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Find MoE layers
moe_layers = []
if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    for layer in model.model.layers:
        if hasattr(layer, 'block_sparse_moe'):
            moe_layers.append(layer.block_sparse_moe)
        elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
            moe_layers.append(layer.mlp)

num_experts = len(moe_layers[0].experts) if moe_layers else 8
tracker = ExpertUsageTracker(num_experts=num_experts, num_layers=len(moe_layers))
radm_margins = {i: [] for i in range(len(moe_layers))}

# Hook to capture expert routing
_routed_debug_printed = False

def _find_selected_from_output(output, num_experts, topk=2):
    if isinstance(output, dict):
        for v in output.values():
            sel = _find_selected_from_output(v, num_experts, topk=topk)
            if sel is not None:
                return sel
        return None
    if isinstance(output, (list, tuple)):
        for v in output:
            sel = _find_selected_from_output(v, num_experts, topk=topk)
            if sel is not None:
                return sel
        return None
    if torch.is_tensor(output):
        if output.dtype in (torch.int32, torch.int64) and output.ndim >= 1 and output.shape[-1] == topk:
            if output.numel() > 0:
                max_val = int(output.max().item())
                min_val = int(output.min().item())
                if 0 <= min_val and max_val < num_experts:
                    return output
    return None

def _find_router_logits(output):
    if isinstance(output, dict):
        for k, v in output.items():
            if "router" in str(k).lower() and torch.is_tensor(v):
                return v
        return None
    if isinstance(output, (list, tuple)) and len(output) >= 2:
        if torch.is_tensor(output[1]):
            return output[1]
    return None

def _compute_selected_from_gate(module, hidden_states, topk=2):
    if hidden_states is None or not torch.is_tensor(hidden_states):
        return None
    if not hasattr(module, "gate"):
        return None
    gate = getattr(module, "gate")
    if not callable(gate):
        return None
    if hidden_states.ndim == 3:
        bsz, seqlen, hdim = hidden_states.shape
        flat = hidden_states.reshape(-1, hdim)
        router_logits = gate(flat)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
        _, selected = torch.topk(routing_weights, k=topk, dim=-1)
        return selected.view(bsz, seqlen, topk)
    if hidden_states.ndim == 2:
        router_logits = gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
        _, selected = torch.topk(routing_weights, k=topk, dim=-1)
        return selected
    return None

def make_hook(layer_idx, model_type):
    """model_type: 'deepseek' or 'default'"""
    def hook(module, input, output):
        global _routed_debug_printed
        if model_type == 'deepseek':
            # DeepSeek: hook the gate directly
            if hasattr(module, 'gate'):
                # Gate returns (topk_idx, topk_weight, aux_loss)
                gate_output = module.gate._last_output if hasattr(module.gate, '_last_output') else None
                if gate_output and isinstance(gate_output, tuple):
                    tracker.log_routing(layer_idx, gate_output[0])
        else:
            # Qwen/Mixtral: MoE returns (output, router_logits)
            selected = None
            if args.routing_source in ('auto', 'indices'):
                selected = _find_selected_from_output(output, num_experts=num_experts, topk=2)
                if selected is None:
                    # Fall back to computing from gate output if available
                    hidden_states = input[0] if input and torch.is_tensor(input[0]) else None
                    selected = _compute_selected_from_gate(module, hidden_states, topk=2)
                if selected is not None and args.routing_debug:
                    if not _routed_debug_printed:
                        print("[EXACT] routing_source=indices (auto)")
                        _routed_debug_printed = True
            if selected is None and args.routing_source in ('auto', 'logits'):
                router_logits = _find_router_logits(output)
                if router_logits is not None:
                    if args.routing_debug:
                        if not _routed_debug_printed:
                            print("[EXACT] routing_source=logits")
                            _routed_debug_printed = True
                    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
                    _, selected = torch.topk(routing_weights, k=2, dim=-1)
            if args.radm:
                router_logits = _find_router_logits(output)
                if router_logits is not None:
                    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
                    top2 = torch.topk(routing_weights, k=2, dim=-1).values
                    margin = (top2[:, 0] - top2[:, 1]).detach().float().cpu().tolist()
                    if margin:
                        buf = radm_margins[layer_idx]
                        if len(buf) < args.radm_max_margins:
                            buf.extend(margin[: args.radm_max_margins - len(buf)])
                        else:
                            import random
                            for m in margin:
                                if len(buf) == 0:
                                    break
                                j = random.randint(0, len(buf) - 1)
                                buf[j] = m
            if selected is not None:
                tracker.log_routing(layer_idx, selected)
    return hook

# Detect model type
model_type = 'deepseek' if 'deepseek' in args.model.lower() else 'default'

# Register hooks
hooks = []
if model_type == 'deepseek':
    # Hook gates directly for DeepSeek
    for i, layer in enumerate(moe_layers):
        def make_gate_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) >= 1:
                    tracker.log_routing(layer_idx, output[0])
            return hook
        hooks.append(layer.gate.register_forward_hook(make_gate_hook(i)))
else:
    # Hook MoE blocks for Qwen/Mixtral
    for i, layer in enumerate(moe_layers):
        hooks.append(layer.register_forward_hook(make_hook(i, model_type)))

# Load dataset
def _run_profile_on_dataset(dataset, text_field, profile_tag=None):
    print(f"Profiling on {args.samples} samples...")
    tracker.enable()
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= args.samples:
                break

            text = sample if text_field is None else sample[text_field]
            if not text or not text.strip():
                continue

            inputs = tokenizer(text, return_tensors='pt', max_length=args.max_length,
                              truncation=True, padding=False).to(model.device)
            model(**inputs)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{args.samples}")

    layer_classifications = {}
    radm_tau_per_layer = {}
    if not args.radm:
        print("\nClassifying experts per layer...")
    for layer_idx in range(len(moe_layers)):
        if not args.radm:
            if args.classify_mode == 'threshold':
                classification = tracker.classify_experts(threshold=args.hot_threshold, layer_idx=layer_idx)
            elif args.classify_mode == 'relative':
                stats = tracker.get_usage_stats(layer_idx)
                if not stats:
                    classification = {'hot': [], 'cold': []}
                else:
                    layer_experts = len(stats)
                    uniform_rate = 1.0 / layer_experts if layer_experts else 0.0
                    threshold = uniform_rate * (1.0 + args.relative_delta)
                    hot = []
                    cold = []
                    for expert_id, data in stats.items():
                        if data['activation_rate'] >= threshold:
                            hot.append(int(expert_id))
                        else:
                            cold.append(int(expert_id))
                    if len(hot) < args.min_hot:
                        items = sorted(
                            ((int(eid), float(d['activation_rate'])) for eid, d in stats.items()),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                        hot = sorted([eid for eid, _ in items[:args.min_hot]])
                        cold = sorted([eid for eid, _ in items[args.min_hot:]])
                    classification = {'hot': sorted(hot), 'cold': sorted(cold)}
            elif args.classify_mode == 'mass':
                stats = tracker.get_usage_stats(layer_idx)
                if not stats:
                    classification = {'hot': [], 'cold': []}
                else:
                    items = sorted(
                        ((int(eid), float(d['activation_rate'])) for eid, d in stats.items()),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    hot = []
                    cold = []
                    cum = 0.0
                    for expert_id, val in items:
                        if cum < args.mass or len(hot) < args.min_hot:
                            hot.append(expert_id)
                            cum += val
                        else:
                            cold.append(expert_id)
                    classification = {'hot': sorted(hot), 'cold': sorted(cold)}
            else:  # percentile
                stats = tracker.get_usage_stats(layer_idx)
                if not stats:
                    classification = {'hot': [], 'cold': []}
                else:
                    values = sorted(float(d['activation_rate']) for d in stats.values())
                    if not values:
                        classification = {'hot': [], 'cold': []}
                    else:
                        p = max(0.0, min(100.0, args.percentile))
                        pos = (p / 100.0) * (len(values) - 1)
                        lo = int(pos)
                        hi = min(lo + 1, len(values) - 1)
                        if hi == lo:
                            threshold = values[lo]
                        else:
                            frac = pos - lo
                            threshold = values[lo] * (1.0 - frac) + values[hi] * frac
                        hot = []
                        cold = []
                        for expert_id, data in stats.items():
                            if data['activation_rate'] >= threshold:
                                hot.append(int(expert_id))
                            else:
                                cold.append(int(expert_id))
                        if len(hot) < args.min_hot:
                            items = sorted(
                                ((int(eid), float(d['activation_rate'])) for eid, d in stats.items()),
                                key=lambda x: x[1],
                                reverse=True,
                            )
                            hot = sorted([eid for eid, _ in items[:args.min_hot]])
                            cold = sorted([eid for eid, _ in items[args.min_hot:]])
                        classification = {'hot': sorted(hot), 'cold': sorted(cold)}
            layer_classifications[layer_idx] = classification
            ent = tracker.get_entropy(layer_idx)
            print(f"Layer {layer_idx}: {len(classification['hot'])} hot, {len(classification['cold'])} cold")
            print(f"  entropy_norm={ent['entropy_norm']:.4f}")
        if args.radm:
            margins = radm_margins.get(layer_idx, [])
            if margins:
                m = torch.tensor(margins, dtype=torch.float32)
                hot_ratio = max(0.0, min(1.0, args.radm_hot_ratio))
                q = 1.0 - hot_ratio
                tau = float(torch.quantile(m, q).item())
            else:
                tau = 0.0
            radm_tau_per_layer[layer_idx] = tau

    model_name = args.model.split('/')[-1].lower()
    output_dir = Path('artifacts/exact_profiling') / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.dataset}" if profile_tag is None else f"{args.dataset}_{profile_tag}"
    if args.radm:
        tag = f"radmhr{args.radm_hot_ratio}"
    elif args.classify_mode == 'threshold':
        tag = f"ht{args.hot_threshold}"
    elif args.classify_mode == 'relative':
        tag = f"reld{args.relative_delta}"
    elif args.classify_mode == 'mass':
        tag = f"mass{args.mass}"
    else:
        tag = f"pct{args.percentile}"
    output_file = output_dir / f'{model_name}_{suffix}_{tag}_real_e{num_experts}_layer.json'

    # Save stats with optional RADM tau
    tracker.save_stats(str(output_file), layer_classifications=layer_classifications)
    if args.radm:
        # Patch in RADM tau per layer to the saved JSON
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            data.setdefault('radm', {})
            data['radm']['tau_per_layer'] = {str(k): float(v) for k, v in radm_tau_per_layer.items()}
            data['radm']['hot_ratio'] = args.radm_hot_ratio
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[EXACT] WARNING: Failed to write RADM tau to JSON: {e}")
    print(f"\n✓ Saved: {output_file}")

    tracker.reset()


print(f"Loading {args.dataset}...")
if args.dataset == 'wikitext':
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    text_field = 'text'
    _run_profile_on_dataset(dataset, text_field)
elif args.dataset == 'c4':
    dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)
    text_field = 'text'
    _run_profile_on_dataset(dataset, text_field)
elif args.dataset == 'eval_tasks':
    def _format_choices(choices):
        if isinstance(choices, dict) and 'text' in choices and 'label' in choices:
            return "\n".join([f"{l}. {t}" for l, t in zip(choices['label'], choices['text'])])
        if isinstance(choices, list):
            return "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        return str(choices)

    def _load_mmlu_astronomy():
        for name, subset in [('hails/mmlu_no_train', 'astronomy'), ('cais/mmlu', 'astronomy'), ('mmlu', 'astronomy')]:
            try:
                return load_dataset(name, subset, split='test')
            except Exception:
                continue
        raise RuntimeError("Unable to load MMLU astronomy dataset. Install or provide dataset name.")

    def _task_iter(task_name):
        if task_name == 'arc_challenge':
            ds = load_dataset('ai2_arc', 'ARC-Challenge', split='test')
            for row in ds:
                q = row.get('question', '')
                c = _format_choices(row.get('choices', {}))
                yield f"{q}\n{c}"
        elif task_name == 'openbookqa':
            ds = load_dataset('openbookqa', 'main', split='test')
            for row in ds:
                q = row.get('question_stem', row.get('question', ''))
                c = _format_choices(row.get('choices', {}))
                yield f"{q}\n{c}"
        elif task_name == 'winogrande':
            ds = load_dataset('winogrande', 'winogrande_xl', split='validation')
            for row in ds:
                sent = row.get('sentence', '')
                opt1 = row.get('option1', '')
                opt2 = row.get('option2', '')
                yield f"{sent}\nA. {opt1}\nB. {opt2}"
        elif task_name == 'gsm8k':
            ds = load_dataset('gsm8k', 'main', split='test')
            for row in ds:
                yield row.get('question', '')
        elif task_name == 'mmlu_astronomy':
            ds = _load_mmlu_astronomy()
            for row in ds:
                q = row.get('question', '')
                c = _format_choices(row.get('choices', row.get('options', [])))
                yield f"{q}\n{c}"
        else:
            raise ValueError(f"Unknown eval task: {task_name}")

    eval_tasks = [t.strip() for t in args.eval_tasks.split(',') if t.strip()]
    if not args.per_task:
        eval_iters = []
        for t in eval_tasks:
            eval_iters.append(_task_iter(t))
        dataset = (text for it in eval_iters for text in it)
        text_field = None
        _run_profile_on_dataset(dataset, text_field)
    else:
        for t in eval_tasks:
            print(f"Profiling task: {t}")
            dataset = (text for text in _task_iter(t))
            text_field = None
            _run_profile_on_dataset(dataset, text_field, profile_tag=t)

# Profile
for hook in hooks:
    hook.remove()
