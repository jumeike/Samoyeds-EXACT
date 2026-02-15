#!/usr/bin/env python3
import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer


def _unwrap_model(m):
    if hasattr(m, "module"):
        return m.module
    if hasattr(m, "_fsdp_wrapped_module"):
        return m._fsdp_wrapped_module
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--sharded_dir", required=True,
                        help="Directory with shard_rank*.pt and sharded_state.txt")
    parser.add_argument("--tasks", default="mmlu_astronomy",
                        help="Comma-separated lm_eval tasks")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--fsdp_min_num_params", type=int, default=10_000_000)
    parser.add_argument("--fsdp_use_orig_params", action="store_true")
    parser.add_argument("--fsdp_offload", action="store_true")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl",
                        help="Process group backend; use gloo to avoid NCCL timeouts")
    parser.add_argument("--distributed_eval", action="store_true",
                        help="Run eval on all ranks using FSDP (avoids full param gather)")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    dist.init_process_group(backend=args.backend)
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    shard_meta = os.path.join(args.sharded_dir, "sharded_state.txt")
    if os.path.exists(shard_meta):
        with open(shard_meta, "r") as f:
            txt = f.read().strip()
        if "world_size=" in txt:
            saved_world_size = int(txt.split("world_size=")[-1].strip())
            if saved_world_size != world_size and rank == 0:
                print(f"WARNING: sharded checkpoint world_size={saved_world_size} "
                      f"but current world_size={world_size}")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.sharded_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    def auto_wrap_policy(module, recurse, nonwrapped_numel):
        return size_based_auto_wrap_policy(
            module=module,
            recurse=recurse,
            nonwrapped_numel=nonwrapped_numel,
            min_num_params=args.fsdp_min_num_params,
        )
    mp_policy = MixedPrecision(param_dtype=torch.float16,
                               reduce_dtype=torch.float16,
                               buffer_dtype=torch.float16)
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        cpu_offload=CPUOffload(offload_params=args.fsdp_offload or args.backend == "gloo"),
        device_id=(local_rank if args.backend == "nccl" else None),
        use_orig_params=args.fsdp_use_orig_params,
    )

    shard_path = os.path.join(args.sharded_dir, f"shard_rank{rank}.pt")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Missing shard file: {shard_path}")
    if rank == 0:
        print(f"Loading sharded checkpoint from {args.sharded_dir} (world_size={world_size})...")
    state_dict = torch.load(shard_path, map_location="cpu")
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        model.load_state_dict(state_dict, strict=True)
    if rank == 0:
        print("✓ Sharded weights loaded")

    from lm_eval import evaluator as lm_evaluator
    from lm_eval.models.huggingface import HFLM

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    model.eval()
    with torch.no_grad():
        if args.distributed_eval:
            # Run eval on all ranks so FSDP comms stay in sync (no full param gather).
            if rank == 0:
                print(f"Running distributed eval on tasks: {', '.join(tasks)}")
            eval_lm = HFLM(pretrained=model, tokenizer=tokenizer)
            results = lm_evaluator.simple_evaluate(
                model=eval_lm,
                tasks=tasks,
                batch_size=args.batch_size,
            )
            if rank == 0:
                for task in tasks:
                    acc = results["results"][task].get("acc,none")
                    acc_norm = results["results"][task].get("acc_norm,none")
                    if acc is not None:
                        print(f"{task}: acc={acc*100:.2f}%")
                    elif acc_norm is not None:
                        print(f"{task}: acc_norm={acc_norm*100:.2f}%")
        else:
            dist.barrier()
            with FSDP.summon_full_params(model, rank0_only=True):
                if rank == 0:
                    eval_model = _unwrap_model(model)
                    if args.backend == "gloo":
                        eval_model = eval_model.to("cuda:0")
                    eval_lm = HFLM(pretrained=eval_model, tokenizer=tokenizer)
                    results = lm_evaluator.simple_evaluate(
                        model=eval_lm,
                        tasks=tasks,
                        batch_size=args.batch_size,
                    )
                    for task in tasks:
                        acc = results["results"][task].get("acc,none")
                        acc_norm = results["results"][task].get("acc_norm,none")
                        if acc is not None:
                            print(f"{task}: acc={acc*100:.2f}%")
                        elif acc_norm is not None:
                            print(f"{task}: acc_norm={acc_norm*100:.2f}%")
                    if args.backend == "gloo":
                        eval_model.to("cpu")
            dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
