"""Benchmark FlashAttention-2 (Triton forward + compiled PyTorch backward) against regular PyTorch attention.

Example:
    uv run python -m cs336_systems.flash_benchmark --output results/data/4_2_flash_benchmarking.jsonl

Sweeps seq_len x d_model x dtype with batch size 1 and causal masking, as in the flash_benchmarking problem. Two
impls: "pytorch" is cs336_basics' `scaled_dot_product_attention` with a precomputed lower-triangular bool mask (the
4.1 baseline), and "flash" is `FlashAttentionTriton.apply(Q, K, V, True)` with autotuned tiles.

For each config, inputs (Q, K, V of shape (1, seq_len, d_model), requiring grad, and an upstream gradient dO) are
generated up front. Three `triton.testing.do_bench` medians (ms) are then taken, each stage after an untimed warm-up
call, so neither the Triton autotuning nor torch.compile of the backward is timed. The forward is warmed up and timed
before any backward runs, so a backward that runs out of memory still leaves a forward time:
    forward:  attn(Q, K, V)
    backward: out.backward(dO, retain_graph=True) on one saved forward output
    end_to_end: attn(Q, K, V).backward(dO)
An out-of-memory error is recorded as "oom_<stage>" (the stages measured before it are kept) and the grid continues.
`torch._dynamo.reset()` runs before each config, so the compiled backward gets a fresh graph per shape instead of
hitting the recompile limit.
"""

import argparse
import gc
import itertools
import json

import torch
import triton.testing

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attention import FlashAttentionTriton, flash_fwd_kernel_autotuned

IMPLS = ("pytorch", "flash")
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[2**i for i in range(7, 17)])
    p.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--dtypes", choices=DTYPES, nargs="+", default=list(DTYPES))
    p.add_argument("--impls", choices=IMPLS, nargs="+", default=list(IMPLS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", help="append one JSON line per config to this file")
    return p.parse_args(argv)


def bench_config(impl: str, seq_len: int, d_model: int, dtype: torch.dtype) -> dict:
    result = dict(impl=impl, batch_size=1, seq_len=seq_len, d_model=d_model, dtype=str(dtype).removeprefix("torch."), causal=True)
    stage = "setup"
    try:
        Q, K, V = (torch.randn(1, seq_len, d_model, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3))
        dO = torch.randn(1, seq_len, d_model, device="cuda", dtype=dtype)
        inputs = [Q, K, V]
        if impl == "pytorch":
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda"))
            attn = lambda: scaled_dot_product_attention(Q, K, V, mask)  # noqa: E731
        else:
            attn = lambda: FlashAttentionTriton.apply(Q, K, V, True)  # noqa: E731

        def bench(fn) -> float:
            return triton.testing.do_bench(fn, grad_to_none=inputs, return_mode="median")

        stage = "forward"
        attn()
        if impl == "flash":
            cfg = flash_fwd_kernel_autotuned.best_config
            result["tile_config"] = dict(**cfg.kwargs, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
        result["forward_ms"] = bench(attn)

        stage = "backward"
        out = attn()
        out.backward(dO, retain_graph=True)
        result["backward_ms"] = bench(lambda: out.backward(dO, retain_graph=True))
        out = None

        stage = "end_to_end"
        result["end_to_end_ms"] = bench(lambda: attn().backward(dO))
        result["status"] = "ok"
    except torch.OutOfMemoryError:
        result["status"] = f"oom_{stage}"
    finally:
        # Drop every tensor (including a retained graph) before the next config.
        Q = K = V = dO = mask = inputs = out = attn = None  # noqa: F841
        gc.collect()
        torch.cuda.empty_cache()
    return result


def format_row(r: dict) -> str:
    head = f"{r['impl']:>7} {r['dtype']:>8} d={r['d_model']:>3} seq={r['seq_len']:>5}"
    times = " | ".join(
        f"{stage} {r[f'{stage}_ms']:9.3f} ms" if f"{stage}_ms" in r else f"{stage} {'-':>12}"
        for stage in ("forward", "backward", "end_to_end")
    )
    tiles = f" | tiles {r['tile_config']}" if "tile_config" in r else ""
    status = "" if r["status"] == "ok" else f" | {r['status'].upper()}"
    return f"{head} | {times}{status}{tiles}"


def main(argv: list[str] | None = None) -> list[dict]:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    print(f"device={torch.cuda.get_device_name()} impls={args.impls} dtypes={args.dtypes}")
    results = []
    for dtype_name, d_model, seq_len, impl in itertools.product(args.dtypes, sorted(args.d_models), sorted(args.seq_lens), args.impls):
        torch._dynamo.reset()
        r = bench_config(impl, seq_len, d_model, DTYPES[dtype_name])
        print(format_row(r), flush=True)
        results.append(r)
        if args.output:
            with open(args.output, "a") as f:
                f.write(json.dumps(r) + "\n")
    return results


if __name__ == "__main__":
    main()
