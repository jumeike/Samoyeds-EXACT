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
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--sharded_dir", required=True,
                        help="Directory with shard_rank*.pt and sharded_state.txt")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save full HF checkpoint")
    parser.add_argument("--fsdp_min_num_params", type=int, default=10_000_000)
    parser.add_argument("--fsdp_use_orig_params", action="store_true")
    parser.add_argument("--fsdp_offload", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    shard_meta = os.path.join(args.sharded_dir, "sharded_state.txt")
    if os.path.exists(shard_meta) and rank == 0:
        with open(shard_meta, "r") as f:
            meta = f.read().strip()
        if "world_size=" in meta:
            saved_world = int(meta.split("world_size=")[-1].strip())
            if saved_world != world_size:
                print(f"WARNING: sharded checkpoint world_size={saved_world} "
                      f"but current world_size={world_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.sharded_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    )

    def auto_wrap_policy(module, recurse, nonwrapped_numel):
        return size_based_auto_wrap_policy(
            module=module,
            recurse=recurse,
            nonwrapped_numel=nonwrapped_numel,
            min_num_params=args.fsdp_min_num_params,
        )
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
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

    shard_path = os.path.join(args.sharded_dir, f"shard_rank{rank}.pt")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Missing shard file: {shard_path}")
    if rank == 0:
        print(f"Loading sharded checkpoint from {args.sharded_dir} (world_size={world_size})...")
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = torch.load(shard_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    if rank == 0:
        print("Gathering full state dict to rank0 (may take time)...")
    full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        full_state = model.state_dict()
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        model_to_save = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        )
        model_to_save.load_state_dict(full_state, strict=True)
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Saved full checkpoint to {args.output_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
