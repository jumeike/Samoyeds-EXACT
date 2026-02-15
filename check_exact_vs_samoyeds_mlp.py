#!/usr/bin/env python3
import argparse
from copy import deepcopy

import numpy as np
import torch

from deepseek.configuration_deepseek import DeepseekConfig
from deepseek.modeling_deepseek import DeepseekMoE
from deepseek.modeling_deepseek_exact import deepseek_moe_to_exact, deepseek_moe_to_ss

from mixtral.configuration_mixtral import MixtralConfig
from mixtral.modeling_mixtral import MixtralSparseMoeBlock
from mixtral.modeling_mixtral_exact import sparsemoeblock_to_exact
from mixtral.modeling_mixtral_samoyeds import sparsemoeblock_to_ss

from qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from qwen2_moe.modeling_qwen2_moe_exact import qwen2_moe_to_exact, qwen2_moe_to_ss
from exact_core.samoyed_baseline_utils import apply_uniform_sparsification


def compare_tensors(a: torch.Tensor, b: torch.Tensor, name: str) -> tuple[float, float]:
    a32 = a.detach().float().cpu()
    b32 = b.detach().float().cpu()
    abs_diff = (a32 - b32).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    flat = abs_diff.view(-1)
    argmax = int(flat.argmax().item())
    max_idx = np.unravel_index(argmax, abs_diff.shape)
    count_1e2 = int((abs_diff > 1e-2).sum().item())
    count_1e1 = int((abs_diff > 1e-1).sum().item())
    count_5e1 = int((abs_diff > 5e-1).sum().item())
    print(f"[{name}] max|diff|={max_abs:.6f}, mean|diff|={mean_abs:.6f}, shape={tuple(a32.shape)}")
    print(
        f"[{name}] >1e-2={count_1e2}, >1e-1={count_1e1}, >5e-1={count_5e1}, "
        f"argmax_idx={tuple(int(x) for x in max_idx)}, "
        f"a={a32[max_idx].item():.6f}, b={b32[max_idx].item():.6f}"
    )
    return max_abs, mean_abs


def compare_packed_expert(sam_model, exact_model, tag: str):
    sam_exp = sam_model.experts[0]
    ex_exp = exact_model.experts[0]
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        s = getattr(sam_exp, proj)
        e = getattr(ex_exp, proj)
        w_eq = torch.equal(s.weight.detach().cpu(), e.weight.detach().cpu())
        i_eq = torch.equal(s.indices.detach().cpu(), e.indices.detach().cpu())
        m_eq = torch.equal(s.metadata.detach().cpu(), e.metadata.detach().cpu())
        w_diff = (s.weight.detach().float().cpu() - e.weight.detach().float().cpu()).abs().max().item()
        print(
            f"[{tag}] expert0.{proj}: weight_eq={w_eq} indices_eq={i_eq} metadata_eq={m_eq} "
            f"max|weight diff|={w_diff:.6f}"
        )


def repeat_stability_check(model, inp, name: str):
    with torch.no_grad():
        o1 = model(inp)
        torch.cuda.synchronize()
        o2 = model(inp)
        torch.cuda.synchronize()
    if isinstance(o1, tuple):
        o1 = o1[0]
    if isinstance(o2, tuple):
        o2 = o2[0]
    compare_tensors(o1, o2, f"{name} repeat")


def copy_model_state_and_nm(src_model, dst_model):
    # Copy all matching params/buffers (includes sparse packed tensors).
    dst_model.load_state_dict(src_model.state_dict(), strict=False)
    # Copy N/M attributes for sparse linear wrappers.
    src_modules = dict(src_model.named_modules())
    for name, dst_mod in dst_model.named_modules():
        src_mod = src_modules.get(name, None)
        if src_mod is None:
            continue
        if hasattr(src_mod, "N") and hasattr(dst_mod, "N"):
            dst_mod.N = src_mod.N
        if hasattr(src_mod, "M") and hasattr(dst_mod, "M"):
            dst_mod.M = src_mod.M


def build_deepseek(args):
    cfg = DeepseekConfig(
        vocab_size=102400,
        hidden_size=args.hidden_size,
        intermediate_size=10944,
        moe_intermediate_size=args.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=16,
        n_shared_experts=2,
        n_routed_experts=args.experts,
        num_experts_per_tok=6,
        moe_layer_freq=1,
        first_k_dense_replace=0,
        norm_topk_prob=False,
        hidden_act="silu",
    )
    dense = DeepseekMoE(cfg)
    sam = deepseek_moe_to_ss(deepcopy(dense))
    apply_uniform_sparsification(sam, dense)
    sam = sam.half().cuda().eval()
    ex = deepseek_moe_to_exact(deepcopy(dense), args.profiling_json).half().cuda().eval()
    if args.force_shared_packed:
        copy_model_state_and_nm(sam, ex)
    inp = torch.randn(
        args.batch_size, args.seq_len, args.hidden_size, device="cuda", dtype=torch.float16
    )
    with torch.no_grad():
        out_s = out_e = None
        if args.run in ("both", "samoyeds"):
            out_s = sam(inp)
            torch.cuda.synchronize()
        if args.run in ("both", "exact"):
            out_e = ex(inp)
            torch.cuda.synchronize()
    return out_s, out_e, sam, ex


def build_mixtral(args):
    cfg = MixtralConfig(
        vocab_size=32000,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=4096 * 32,
        rms_norm_eps=1e-5,
        rope_theta=1e6,
        sliding_window=4096,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_local_experts=args.experts,
        output_router_logits=False,
    )
    dense = MixtralSparseMoeBlock(cfg)
    sam = sparsemoeblock_to_ss(deepcopy(dense), skip_sparsifier=True)
    apply_uniform_sparsification(sam, dense)
    sam = sam.half().cuda().eval()
    ex = sparsemoeblock_to_exact(deepcopy(dense), args.profiling_json).half().cuda().eval()
    if args.force_shared_packed:
        copy_model_state_and_nm(sam, ex)
    inp = torch.randn(
        args.batch_size, args.seq_len, args.hidden_size, device="cuda", dtype=torch.float16
    )
    with torch.no_grad():
        out_s = out_e = None
        if args.run in ("both", "samoyeds"):
            out_s = sam(inp)
            torch.cuda.synchronize()
        if args.run in ("both", "exact"):
            out_e = ex(inp)
            torch.cuda.synchronize()
    return (out_s[0] if out_s is not None else None), (out_e[0] if out_e is not None else None), sam, ex


def build_qwen(args):
    cfg = Qwen2MoeConfig(
        vocab_size=151936,
        hidden_size=args.hidden_size,
        intermediate_size=5632,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        moe_intermediate_size=args.intermediate_size,
        shared_expert_intermediate_size=5632,
        num_experts_per_tok=4,
        num_experts=args.experts,
        norm_topk_prob=False,
        output_router_logits=False,
    )
    dense = Qwen2MoeSparseMoeBlock(cfg)
    sam = qwen2_moe_to_ss(deepcopy(dense))
    apply_uniform_sparsification(sam, dense)
    sam = sam.half().cuda().eval()
    ex = qwen2_moe_to_exact(deepcopy(dense), args.profiling_json).half().cuda().eval()
    if args.force_shared_packed:
        copy_model_state_and_nm(sam, ex)
    inp = torch.randn(
        args.batch_size, args.seq_len, args.hidden_size, device="cuda", dtype=torch.float16
    )
    with torch.no_grad():
        out_s = out_e = None
        if args.run in ("both", "samoyeds"):
            out_s = sam(inp)
            torch.cuda.synchronize()
        if args.run in ("both", "exact"):
            out_e = ex(inp)
            torch.cuda.synchronize()
    return (out_s[0] if out_s is not None else None), (out_e[0] if out_e is not None else None), sam, ex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["deepseek", "mixtral", "qwen2_moe"], required=True)
    parser.add_argument("--profiling-json", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--experts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-abs-thr", type=float, default=2e-1)
    parser.add_argument("--run", choices=["both", "samoyeds", "exact"], default="both")
    parser.add_argument("--debug-packed", action="store_true")
    parser.add_argument("--force-shared-packed", action="store_true")
    parser.add_argument("--stability-check", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    if args.model == "deepseek":
        args.hidden_size = args.hidden_size or 2048
        args.intermediate_size = args.intermediate_size or 1408
        args.experts = args.experts or 8
        if args.seq_len < 512:
            print(
                "[warn] DeepSeek Samoyeds path can fault for very small token counts. "
                "Use benchmark-like seq_len (e.g., 4096)."
            )
        out_s, out_e, sam_model, exact_model = build_deepseek(args)
        probe_inp = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device="cuda", dtype=torch.float16)
    elif args.model == "mixtral":
        args.hidden_size = args.hidden_size or 4096
        args.intermediate_size = args.intermediate_size or 14336
        args.experts = args.experts or 8
        out_s, out_e, sam_model, exact_model = build_mixtral(args)
        probe_inp = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device="cuda", dtype=torch.float16)
    else:
        args.hidden_size = args.hidden_size or 2048
        args.intermediate_size = args.intermediate_size or 5632
        args.experts = args.experts or 60
        out_s, out_e, sam_model, exact_model = build_qwen(args)
        probe_inp = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device="cuda", dtype=torch.float16)

    if args.run == "samoyeds":
        print(f"[result] PASS: Samoyeds-only forward succeeded for {args.model}")
        return 0
    if args.run == "exact":
        print(f"[result] PASS: EXACT-only forward succeeded for {args.model}")
        return 0
    if args.debug_packed:
        compare_packed_expert(sam_model, exact_model, args.model)
    if args.stability_check:
        repeat_stability_check(sam_model, probe_inp, f"{args.model} samoyeds")
        repeat_stability_check(exact_model, probe_inp, f"{args.model} exact")

    max_abs, _ = compare_tensors(out_s, out_e, f"{args.model} Samoyeds vs EXACT")
    if max_abs > args.max_abs_thr:
        print(f"[result] FAIL: max|diff| {max_abs:.6f} > threshold {args.max_abs_thr:.6f}")
        return 1
    print(f"[result] PASS: max|diff| {max_abs:.6f} <= threshold {args.max_abs_thr:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
