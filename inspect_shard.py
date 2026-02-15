import os, torch, torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()

path=f'artifacts/finetuned_exact_mmlu_2k/shard_rank{rank}.pt'
print('rank', rank, 'exists', os.path.exists(path))
state=torch.load(path, map_location='cpu')
if rank == 0:
    print(type(state))
    print(list(state.keys())[:5])
    first_key=list(state.keys())[0]
    print('first key', first_key, type(state[first_key]))
dist.destroy_process_group()
