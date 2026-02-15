#!/usr/bin/env python3
"""
Finetune HF Qwen MoE with fixed EXACT masks (hot 2:4, cold 1:4).
This script preserves sparsity by re-applying masks after each optimizer step.
"""
import os
os.environ['HF_HOME'] = '/fs/nexus-projects/EXACT/EXACT-Project/checkpoints/hf_cache'

import argparse
import json
import os
import time
import functools
import types
import importlib
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.optimization import Adafactor
from datasets import load_dataset
from exact_core.samoyed_converter import _prune_24_magnitude, _prune_14_magnitude
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


def _get_layer_expert_path_and_container(layer):
    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
        return 'mlp.experts', layer.mlp.experts
    if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'experts'):
        return 'block_sparse_moe.experts', layer.block_sparse_moe.experts
    return None, None


def _get_expert_proj_names(expert):
    qwen_deepseek = [n for n in ['gate_proj', 'up_proj', 'down_proj'] if hasattr(expert, n)]
    if qwen_deepseek:
        return qwen_deepseek
    mixtral = [n for n in ['w1', 'w2', 'w3'] if hasattr(expert, n)]
    return mixtral


def build_masks(model, profiling_json, cold_as_24=False, name_prefix=""):
    with open(profiling_json) as f:
        prof = json.load(f)
    if 'layers' not in prof:
        raise ValueError(f"Expected 'layers' key. Got: {list(prof.keys())}")
    layers_data = prof['layers']

    masks = {}
    for l_idx, layer in enumerate(model.model.layers):
        experts_path, experts = _get_layer_expert_path_and_container(layer)
        if experts is None:
            continue
        layer_key = str(l_idx)
        classification = layers_data.get(layer_key, {}).get('classification', {})
        hot = set(classification.get('hot', []))
        cold = set(classification.get('cold', []))

        for e_idx, expert in _iter_experts(experts):
            if e_idx in hot:
                pruner = _prune_24_magnitude
            elif e_idx in cold:
                pruner = _prune_24_magnitude if cold_as_24 else _prune_14_magnitude
            else:
                pruner = _prune_24_magnitude
            proj_names = _get_expert_proj_names(expert)
            if not proj_names:
                continue
            for name in proj_names:
                linear = getattr(expert, name)
                pruned_cpu = pruner(linear.weight.data, return_to_device=False)
                pruned = pruned_cpu.to(linear.weight.device, dtype=linear.weight.dtype)
                linear.weight.data.copy_(pruned)
                mask = (pruned != 0).to(linear.weight.device)
                param_name = f"{name_prefix}model.layers.{l_idx}.{experts_path}.{e_idx}.{name}.weight"
                masks[param_name] = mask.detach().cpu()
    return masks


def get_layer_experts(layer):
    _, experts = _get_layer_expert_path_and_container(layer)
    if experts is not None:
        return experts
    return None


def _iter_experts(experts):
    # Support ModuleDict/OrderedDict (items) and ModuleList/Sequential (enumerate).
    if hasattr(experts, "items"):
        for k, v in experts.items():
            eid = None
            if isinstance(k, str):
                if k.isdigit():
                    eid = int(k)
                else:
                    parts = k.split("_")
                    if parts and parts[-1].isdigit():
                        eid = int(parts[-1])
            if eid is None:
                # Fallback to enumeration index when key is not parseable.
                eid = 0
            yield eid, v
    else:
        for idx, v in enumerate(experts):
            yield idx, v


def _unwrap_model(m):
    if hasattr(m, "module"):
        return m.module
    if hasattr(m, "_fsdp_wrapped_module"):
        return m._fsdp_wrapped_module
    return m


def _patch_deepseek_moe_forward_dtype(model):
    """
    DeepSeek MoE implementation may assign expert outputs into an intermediate
    tensor with a different dtype. Under fp16/bf16 this can raise:
    "Index put requires source and destination dtypes match".
    Patch DeepseekMoE.forward so assignment always uses y.dtype.
    """
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekMoE":
            continue
        if getattr(module, "_exact_moe_forward_patch_applied", False):
            continue

        orig_forward = module.forward
        module_obj = importlib.import_module(module.__class__.__module__)
        add_aux_cls = getattr(module_obj, "AddAuxiliaryLoss", None)

        def _wrapped_forward(self, hidden_states, __orig=orig_forward, __add_aux=add_aux_cls):
            identity = hidden_states
            orig_shape = hidden_states.shape
            topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
            flat_topk_idx = topk_idx.view(-1)
            if self.training:
                hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
                y = torch.empty_like(hidden_states)
                for i, expert in enumerate(self.experts):
                    idx = (flat_topk_idx == i)
                    if idx.any():
                        out = expert(hidden_states[idx])
                        if torch.is_tensor(out) and out.dtype != y.dtype:
                            out = out.to(y.dtype)
                        y[idx] = out
                y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
                y = y.view(*orig_shape)
                if __add_aux is not None:
                    y = __add_aux.apply(y, aux_loss)
            else:
                y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
            if self.config.n_shared_experts is not None:
                shared = self.shared_experts(identity)
                if torch.is_tensor(shared) and shared.dtype != y.dtype:
                    shared = shared.to(y.dtype)
                y = y + shared
            return y

        module.forward = types.MethodType(_wrapped_forward, module)
        module._exact_moe_forward_patch_applied = True
        patched += 1
    return patched


def set_trainable_params(model, profiling_json, mode="experts", train_shared_expert=False, train_gate=False):
    """
    mode: 'all', 'experts', 'hot', 'cold'
    """
    model = _unwrap_model(model)
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    with open(profiling_json) as f:
        prof = json.load(f)
    layers_data = prof.get('layers', {})

    # Default: freeze everything
    for p in model.parameters():
        p.requires_grad = False

    for l_idx, layer in enumerate(model.model.layers):
        experts = get_layer_experts(layer)
        if experts is None:
            continue
        cls = layers_data.get(str(l_idx), {}).get('classification', {})
        hot = set(cls.get('hot', []))
        cold = set(cls.get('cold', []))
        for e_idx, expert in _iter_experts(experts):
            if mode == "experts":
                allow = True
            elif mode == "hot":
                allow = e_idx in hot
            elif mode == "cold":
                allow = e_idx in cold
            else:
                allow = False
            if not allow:
                continue
            for name in ['gate_proj', 'up_proj', 'down_proj', 'w1', 'w2', 'w3']:
                if hasattr(expert, name):
                    for p in getattr(expert, name).parameters():
                        p.requires_grad = True
        if train_shared_expert and hasattr(layer, 'mlp') and hasattr(layer.mlp, 'shared_expert'):
            for p in layer.mlp.shared_expert.parameters():
                p.requires_grad = True
        if train_gate and hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            for p in layer.mlp.gate.parameters():
                p.requires_grad = True


def apply_masks(model, masks, name_prefix=""):
    if masks is None:
        return 0, 0
    def _candidate_names(param_name: str):
        # Raw name
        yield param_name
        # Strip common wrapper prefixes once
        for pfx in ("module.", "_fsdp_wrapped_module."):
            if param_name.startswith(pfx):
                yield param_name[len(pfx):]
        # Strip repeated wrapper prefixes (e.g., module._fsdp_wrapped_module.*)
        cur = param_name
        changed = True
        while changed:
            changed = False
            for pfx in ("module.", "_fsdp_wrapped_module."):
                if cur.startswith(pfx):
                    cur = cur[len(pfx):]
                    changed = True
        if cur != param_name:
            yield cur

    matched = set()
    for name, param in model.named_parameters():
        if not name.startswith(name_prefix):
            continue
        match = None
        for cand in _candidate_names(name):
            if cand in masks:
                match = cand
                break
        if match is not None:
            matched.add(match)
            mask = masks[match].to(param.device, dtype=param.dtype)
            param.data.mul_(mask)
            # Check for NaN or Inf after masking
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                print(f"WARNING: NaN/Inf detected in {name} after applying mask")
    return len(matched), len(masks)


def get_dataset_iterator(tokenizer, max_length, dataset, dataset_config, split,
                         rank=0, world_size=1, cache_dir=None, data_dir=None,
                         offline=False):
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    ds = load_dataset(
        dataset,
        dataset_config,
        split=split,
        streaming=True,
        cache_dir=cache_dir,
        data_dir=data_dir,
        download_mode="reuse_dataset_if_exists",
    )
    for idx, row in enumerate(ds):
        if world_size > 1 and (idx % world_size) != rank:
            continue
        text = row.get('text', '')
        if not text or not text.strip():
            continue
        tokens = tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding='max_length',
        )
        yield tokens['input_ids'], tokens['attention_mask']


def get_eval_tasks_iterator(tokenizer, max_length, tasks, rank=0, world_size=1):
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
        raise RuntimeError("Unable to load MMLU astronomy dataset.")

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

    tasks = [t.strip() for t in tasks.split(',') if t.strip()]
    sample_idx = 0
    for t in tasks:
        for text in _task_iter(t):
            if world_size > 1 and (sample_idx % world_size) != rank:
                sample_idx += 1
                continue
            if not text or not text.strip():
                sample_idx += 1
                continue
            tokens = tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding='max_length',
            )
            sample_idx += 1
            yield tokens['input_ids'], tokens['attention_mask']


def infinite_iterator(factory_fn):
    while True:
        for item in factory_fn():
            yield item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen1.5-MoE-A2.7B')
    parser.add_argument('--profiling_json', required=True)
    parser.add_argument('--cold_as_24', action='store_true')
    parser.add_argument('--dataset', default='allenai/c4',
                        help='HF dataset name (default: allenai/c4)')
    parser.add_argument('--dataset_config', default='en',
                        help='HF dataset config (default: en)')
    parser.add_argument('--dataset_split', default='train',
                        help='HF dataset split (default: train)')
    parser.add_argument('--dataset_cache_dir', default=os.environ.get('HF_HOME'),
                        help='HF datasets cache dir (default: HF_HOME)')
    parser.add_argument('--dataset_data_dir', default=None,
                        help='Optional local data_dir for dataset')
    parser.add_argument('--dataset_offline', action='store_true',
                        help='Force offline mode (use local cache only)')
    parser.add_argument('--eval_tasks', default='mmlu_astronomy',
                        help='Comma-separated eval tasks when --dataset eval_tasks')
    parser.add_argument('--eval_every', type=int, default=0,
                        help='Run periodic MMLU eval every N optimizer steps (0 disables)')
    parser.add_argument('--eval_tasks_eval', default='mmlu_astronomy',
                        help='Comma-separated eval tasks for periodic eval (lm_eval)')
    parser.add_argument('--eval_batch_size', type=int, default=1,
                        help='Batch size for periodic eval')
    parser.add_argument('--load_sharded_dir', default=None,
                        help='Load FSDP sharded checkpoint from this directory before training')
    parser.add_argument('--resume_step', type=int, default=0,
                        help='Resume step counter (for logging only)')
    parser.add_argument('--local_mask_on_sharded', action='store_true',
                        help='When resuming from sharded FSDP weights, build/apply masks on local shards (avoids full-param all-gather)')
    parser.add_argument('--save_optimizer', action='store_true',
                        help='Save optimizer state along with model')
    parser.add_argument('--load_optimizer', action='store_true',
                        help='Load optimizer state from load_sharded_dir/save_path if available')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--save_path', default='artifacts/finetuned_exact')
    parser.add_argument('--save_mode', choices=['full', 'sharded', 'none'], default='full',
                        help='FSDP save mode: full gathers all params, sharded saves per-rank, none skips save')
    parser.add_argument('--ddp', action='store_true', help='Enable DDP if launched with torchrun')
    parser.add_argument('--ddp_find_unused_parameters', action='store_true',
                        help='Enable DDP find_unused_parameters (costs memory)')
    parser.add_argument('--ddp_bucket_cap_mb', type=int, default=25,
                        help='DDP bucket size in MB (smaller can reduce peak memory)')
    parser.add_argument('--ddp_gradient_as_bucket_view', action='store_true',
                        help='Use gradient_as_bucket_view to reduce DDP memory')
    parser.add_argument('--ddp_static_graph', action='store_true',
                        help='Enable DDP static_graph to reduce overhead')
    parser.add_argument('--grad_ckpt', action='store_true',
                        help='Enable gradient checkpointing to reduce memory')
    parser.add_argument('--optimizer', choices=['adamw', 'adafactor'], default='adamw',
                        help='Optimizer choice (adafactor uses less memory)')
    parser.add_argument('--amp_dtype', choices=['fp16', 'bf16', 'fp32'], default='fp16',
                        help='Autocast dtype (fp32 disables autocast)')
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='Gradient clipping max norm (0 disables)')
    parser.add_argument('--mask_every', type=int, default=1,
                        help='Reapply masks every N optimizer steps')
    parser.add_argument('--fsdp', action='store_true', help='Enable FSDP sharding')
    parser.add_argument('--fsdp_offload', action='store_true', help='Enable FSDP CPU offload')
    parser.add_argument('--fsdp_min_num_params', type=int, default=10_000_000,
                        help='FSDP auto-wrap threshold')
    parser.add_argument('--fsdp_use_orig_params', action='store_true',
                        help='Use FSDP original parameters (avoids flattening)')
    parser.add_argument('--train_mode', choices=['all', 'experts', 'hot', 'cold'], default='experts',
                        help='Which parameters to finetune')
    parser.add_argument('--train_shared_expert', action='store_true',
                        help='Also finetune shared expert if present')
    parser.add_argument('--train_gate', action='store_true',
                        help='Also finetune gate if present')
    args = parser.parse_args()

    use_ddp = (args.ddp or int(os.environ.get("WORLD_SIZE", "1")) > 1) and not args.fsdp
    use_fsdp = args.fsdp
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if use_ddp or use_fsdp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.amp_dtype == 'bf16':
        model_dtype = torch.bfloat16
    elif args.amp_dtype == 'fp32':
        model_dtype = torch.float32
    else:
        model_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        device_map=None,
        trust_remote_code=True,
    )
    if "deepseek" in args.model.lower():
        patched = _patch_deepseek_moe_forward_dtype(model)
        if rank == 0:
            print(f"[deepseek] patched expert dtype adapters: {patched}")
    if not use_fsdp:
        model = model.to(device)
    if args.grad_ckpt:
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    model.train()

    # Set trainable params before wrapping with FSDP/DDP to avoid losing requires_grad.
    set_trainable_params(
        model,
        args.profiling_json,
        mode=args.train_mode,
        train_shared_expert=args.train_shared_expert,
        train_gate=args.train_gate,
    )
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError(
            "No trainable parameters were enabled. "
            "Check --train_mode and profiling_json alignment."
        )

    masks = None

    if use_fsdp:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=args.fsdp_min_num_params
        )
        mp_policy = MixedPrecision(
            param_dtype=model_dtype,
            reduce_dtype=model_dtype,
            buffer_dtype=model_dtype
        )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            cpu_offload=CPUOffload(offload_params=args.fsdp_offload),
            device_id=local_rank,
            use_orig_params=args.fsdp_use_orig_params,
        )
        if args.load_sharded_dir:
            shard_meta = os.path.join(args.load_sharded_dir, "sharded_state.txt")
            if os.path.exists(shard_meta) and rank == 0:
                with open(shard_meta, "r") as f:
                    meta = f.read().strip()
                if "world_size=" in meta:
                    saved_world = int(meta.split("world_size=")[-1].strip())
                    if saved_world != world_size:
                        print(f"WARNING: sharded checkpoint world_size={saved_world} "
                              f"but current world_size={world_size}")
            shard_path = os.path.join(args.load_sharded_dir, f"shard_rank{rank}.pt")
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Missing shard file: {shard_path}")
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                state_dict = torch.load(shard_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
        # Build masks once using full params to avoid flattened shards.
        if args.load_sharded_dir and args.local_mask_on_sharded:
            # Avoid full-param all-gather on resume; build masks on local shards.
            masks = build_masks(model, args.profiling_json, cold_as_24=args.cold_as_24)
            matched, total = apply_masks(model, masks)
            if rank == 0:
                print(f"[mask] initial apply matched {matched}/{total}")
        else:
            with FSDP.summon_full_params(model):
                masks = build_masks(model, args.profiling_json, cold_as_24=args.cold_as_24)
                # Apply masks initially to ensure sparsity from the start
                matched, total = apply_masks(model, masks)
                if rank == 0:
                    print(f"[mask] initial apply matched {matched}/{total}")
    elif use_ddp:
        # Build and apply masks before wrapping with DDP
        masks = build_masks(model, args.profiling_json, cold_as_24=args.cold_as_24)
        matched, total = apply_masks(model, masks)
        if rank == 0:
            print(f"[mask] initial apply matched {matched}/{total}")
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.ddp_find_unused_parameters,
            bucket_cap_mb=args.ddp_bucket_cap_mb,
            gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
            static_graph=args.ddp_static_graph,
        )
    else:
        # Build and apply masks for single GPU training
        masks = build_masks(model, args.profiling_json, cold_as_24=args.cold_as_24)
        matched, total = apply_masks(model, masks)
        print(f"[mask] initial apply matched {matched}/{total}")

    # Trainable params already set before wrapping

    if args.optimizer == 'adafactor':
        optimizer = Adafactor(
            model.parameters(),
            lr=args.lr,
            scale_parameter=False,
            relative_step=False
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.load_optimizer:
        opt_dir = args.load_sharded_dir or args.save_path
        opt_path = os.path.join(opt_dir, f"optimizer_rank{rank}.pt") if use_fsdp or use_ddp else os.path.join(opt_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            if rank == 0:
                print(f"Loaded optimizer state from {opt_path}")
        else:
            if rank == 0:
                print(f"Optimizer state not found at {opt_path}; starting fresh.")
    amp_dtype = None
    if args.amp_dtype == 'bf16':
        amp_dtype = torch.bfloat16
    elif args.amp_dtype == 'fp16':
        amp_dtype = torch.float16
    # FSDP uses FP16 params; GradScaler expects FP32 grads, so disable under FSDP.
    # Also disable GradScaler for bfloat16 (bf16 doesn't need gradient scaling)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == 'cuda' and not use_fsdp and args.amp_dtype == 'fp16')
    )

    def _make_data_iter():
        if args.dataset == 'eval_tasks':
            return get_eval_tasks_iterator(
                tokenizer,
                args.max_length,
                args.eval_tasks,
                rank=rank,
                world_size=world_size,
            )
        return get_dataset_iterator(
            tokenizer,
            args.max_length,
            args.dataset,
            args.dataset_config,
            args.dataset_split,
            rank=rank,
            world_size=world_size,
            cache_dir=args.dataset_cache_dir,
            data_dir=args.dataset_data_dir,
            offline=args.dataset_offline,
        )

    try:
        data_iter = infinite_iterator(_make_data_iter)
    except Exception as exc:
        if rank == 0:
            print(f"Failed to load dataset {args.dataset}:{args.dataset_config} -> {exc}")
            print("Tip: try --dataset wikitext --dataset_config wikitext-2-raw-v1 "
                  "or set --dataset_offline if you have a local cache.")
        raise

    if args.eval_every and args.eval_every > 0:
        from lm_eval import evaluator as lm_evaluator
        from lm_eval.models.huggingface import HFLM

    step = args.resume_step
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    while step < args.steps:
        batch_ids = []
        batch_attn = []
        for _ in range(args.batch_size):
            input_ids, attn = next(data_iter)
            batch_ids.append(input_ids)
            batch_attn.append(attn)
        input_ids = torch.cat(batch_ids, dim=0).to(device)
        attention_mask = torch.cat(batch_attn, dim=0).to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda' and args.amp_dtype != 'fp32'),
                                     dtype=amp_dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % args.grad_accum == 0:
            if args.grad_clip and args.grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                if use_fsdp:
                    # Fast path: apply masks on local shards during training.
                    # Full-param correctness is still enforced before full save.
                    if args.mask_every > 0 and (step + 1) // args.grad_accum % args.mask_every == 0:
                        apply_masks(model, masks)
                else:
                    if args.mask_every > 0 and (step + 1) // args.grad_accum % args.mask_every == 0:
                        apply_masks(model, masks)

        if (step + 1) % 100 == 0 and (not use_ddp or rank == 0):
            elapsed = max(time.time() - start_time, 1e-6)
            steps_done = step + 1
            steps_per_sec = steps_done / elapsed
            remaining = max(args.steps - steps_done, 0)
            eta_sec = remaining / max(steps_per_sec, 1e-6)
            eta_min = int(eta_sec // 60)
            eta_s = int(eta_sec % 60)
            # Show unscaled loss (multiply back by grad_accum)
            actual_loss = loss.item() * args.grad_accum
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf detected at step {steps_done}. Loss: {actual_loss}")
            print(
                f"step {steps_done}/{args.steps} loss={actual_loss:.4f} "
                f"({steps_per_sec:.2f} it/s, ETA {eta_min:02d}:{eta_s:02d})"
            )

        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            # Periodic MMLU eval: run on rank 0 only to avoid duplicated work.
            model.eval()
            with torch.no_grad():
                if use_fsdp:
                    # Ensure full params on rank0; all ranks must enter this context.
                    dist.barrier()
                    with FSDP.summon_full_params(model, rank0_only=True):
                        if rank == 0:
                            eval_lm = HFLM(pretrained=_unwrap_model(model), tokenizer=tokenizer)
                            eval_tasks = [t.strip() for t in args.eval_tasks_eval.split(',') if t.strip()]
                            results = lm_evaluator.simple_evaluate(
                                model=eval_lm,
                                tasks=eval_tasks,
                                batch_size=args.eval_batch_size,
                            )
                            for task in eval_tasks:
                                acc = results["results"][task].get("acc,none")
                                acc_norm = results["results"][task].get("acc_norm,none")
                                if acc is not None:
                                    print(f"[eval] step {step + 1}: {task} acc={acc*100:.2f}%")
                                elif acc_norm is not None:
                                    print(f"[eval] step {step + 1}: {task} acc_norm={acc_norm*100:.2f}%")
                    dist.barrier()
                else:
                    if rank == 0:
                        eval_lm = HFLM(pretrained=_unwrap_model(model), tokenizer=tokenizer)
                        eval_tasks = [t.strip() for t in args.eval_tasks_eval.split(',') if t.strip()]
                        results = lm_evaluator.simple_evaluate(
                            model=eval_lm,
                            tasks=eval_tasks,
                            batch_size=args.eval_batch_size,
                        )
                        for task in eval_tasks:
                            acc = results["results"][task].get("acc,none")
                            acc_norm = results["results"][task].get("acc_norm,none")
                            if acc is not None:
                                print(f"[eval] step {step + 1}: {task} acc={acc*100:.2f}%")
                            elif acc_norm is not None:
                                print(f"[eval] step {step + 1}: {task} acc_norm={acc_norm*100:.2f}%")
            model.train()

        step += 1

    if args.save_mode != 'none':
        if not (use_ddp or use_fsdp) or rank == 0:
            os.makedirs(args.save_path, exist_ok=True)
        if use_fsdp:
            if args.save_mode == 'full':
                # Ensure final checkpoint preserves sparsity pattern.
                if masks is not None:
                    with torch.no_grad():
                        with FSDP.summon_full_params(model, offload_to_cpu=True):
                            matched, total = apply_masks(model, masks)
                    if rank == 0:
                        print(f"[mask] pre-save apply matched {matched}/{total}")
                full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
                    state_dict = model.state_dict()
                if rank == 0:
                    # Save directly from gathered state_dict to avoid a second full model
                    # instantiation on CPU (large host-RAM spike for 16B models).
                    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
                    config.save_pretrained(args.save_path)
                    torch.save(state_dict, os.path.join(args.save_path, "pytorch_model.bin"))
            elif args.save_mode == 'sharded':
                # Ensure final sharded checkpoint also preserves sparsity pattern.
                if masks is not None:
                    with torch.no_grad():
                        matched, total = apply_masks(model, masks)
                    if rank == 0:
                        print(f"[mask] pre-save apply matched {matched}/{total}")
                with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                    state_dict = model.state_dict()
                shard_path = os.path.join(args.save_path, f"shard_rank{rank}.pt")
                torch.save(state_dict, shard_path)
                if rank == 0:
                    with open(os.path.join(args.save_path, "sharded_state.txt"), "w") as f:
                        f.write(f"world_size={world_size}\n")
            if rank == 0:
                tokenizer.save_pretrained(args.save_path)
                print(f"Saved finetuned model to {args.save_path} (mode={args.save_mode})")
        else:
            if not (use_ddp or use_fsdp) or rank == 0:
                to_save = model.module if use_ddp else model
                to_save.save_pretrained(args.save_path)
                tokenizer.save_pretrained(args.save_path)
                print(f"Saved finetuned model to {args.save_path}")
        if args.save_optimizer:
            if use_fsdp or use_ddp:
                opt_path = os.path.join(args.save_path, f"optimizer_rank{rank}.pt")
                torch.save(optimizer.state_dict(), opt_path)
            else:
                opt_path = os.path.join(args.save_path, "optimizer.pt")
                torch.save(optimizer.state_dict(), opt_path)
            if rank == 0:
                print(f"Saved optimizer state to {opt_path}")

    if use_ddp or use_fsdp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
