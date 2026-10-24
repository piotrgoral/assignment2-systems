"""Benchmark cs336_basics scaled dot-product attention (single head) over a grid of d_model x seq_len.

Example:
    uv run python -m cs336_systems.attention_benchmark --output results/data/4_1_pytorch_attention.jsonl

For each config: time `--steps` forward passes, measure the memory in use right before backward (inputs plus the
residuals autograd saved), then time `--steps` backward passes (each after an untimed forward, since backward frees
the graph). Every pass is followed by a synchronize. Configs that run out of memory are recorded as
"oom_forward" / "oom_backward" and the grid continues.

Forward and backward passes are wrapped in NVTX ranges ("forward" / "backward"), as in benchmark.py.

`--impls eager compiled` also benchmarks `torch.compile(scaled_dot_product_attention)`, config by config. It is
compiled afresh for every config (after `torch._dynamo.reset()`), so each shape gets its own static graph instead of
piling up recompiles. Compilation happens in the first warm-up pass, whose time is recorded as "first_iter_s".

`--impls flash_pytorch` benchmarks the pure-PyTorch FlashAttention-2 forward (`FlashAttentionPytorch`), with square
tiles of `--flash-tile-size`. It has no backward yet, so only the forward is timed (warm-up runs forward only) and
"mem_before_backward_bytes" is the memory held after a forward pass with its graph alive (Q, K, V, O and L). It
ignores masks, so it cannot be combined with `--causal`.

`--impls flash_triton` does the same for the Triton FlashAttention-2 forward kernel (`FlashAttentionTriton`): square
tiles of `--flash-tile-size`, forward only, no `--causal`. A tile size whose kernel needs more shared memory than the
GPU has is recorded as "out_of_shared_memory" and the grid continues.
"""

import argparse
import gc
import itertools
import json
import timeit
from collections.abc import Callable

import torch
import torch.cuda.nvtx as nvtx
from triton.runtime.errors import OutOfResources

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.benchmark import summarize, synchronize
from cs336_systems.flash_attention import FlashAttentionPytorch, FlashAttentionTriton

AttnFn = Callable[..., torch.Tensor]

IMPLS = ("eager", "compiled", "flash_pytorch", "flash_triton")
FORWARD_ONLY_IMPLS = {"flash_pytorch", "flash_triton"}


def make_attn(impl: str, flash_tile_size: int) -> AttnFn:
    if impl == "eager":
        return scaled_dot_product_attention
    if impl == "flash_pytorch":
        FlashAttentionPytorch.Q_TILE_SIZE = FlashAttentionPytorch.K_TILE_SIZE = flash_tile_size
        return lambda Q, K, V, mask: FlashAttentionPytorch.apply(Q, K, V)
    if impl == "flash_triton":
        FlashAttentionTriton.Q_TILE_SIZE = FlashAttentionTriton.K_TILE_SIZE = flash_tile_size
        return lambda Q, K, V, mask: FlashAttentionTriton.apply(Q, K, V)
    torch._dynamo.reset()
    return torch.compile(scaled_dot_product_attention)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--d-models", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--seq-lens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--impls", choices=IMPLS, nargs="+", default=["eager"], help="attention variants to benchmark")
    p.add_argument("--flash-tile-size", type=int, default=16, help="Bq = Bk for flash_pytorch and flash_triton")
    p.add_argument("--causal", action="store_true", help="pass a lower-triangular (seq_len x seq_len) bool mask")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", help="append one JSON line per config to this file")
    return p.parse_args(argv)


def make_inputs(
    batch_size: int, seq_len: int, d_model: int, causal: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Q, K, V (requiring grad), an optional causal mask, and a fixed upstream gradient for backward."""
    Q, K, V = (torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True) for _ in range(3))
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)) if causal else None
    grad_out = torch.randn(batch_size, seq_len, d_model, device=device)
    return Q, K, V, mask, grad_out


def bench_config(
    attn: AttnFn,
    batch_size: int,
    seq_len: int,
    d_model: int,
    causal: bool,
    warmup_steps: int,
    steps: int,
    device: torch.device,
    run_backward: bool = True,
) -> dict:
    result = dict(batch_size=batch_size, d_model=d_model, seq_len=seq_len, causal=causal, device=str(device))
    stage = "forward"
    try:
        Q, K, V, mask, grad_out = make_inputs(batch_size, seq_len, d_model, causal, device)
        inputs = (Q, K, V)

        def backward(out: torch.Tensor) -> None:
            out.backward(grad_out)
            for x in inputs:
                x.grad = None

        # Warm-up runs backward too, so track the stage within it: an OOM there is reported as the pass that hit it.
        for i in range(warmup_steps):
            start = timeit.default_timer()
            stage = "forward"
            out = attn(Q, K, V, mask)
            if run_backward:
                stage = "backward"
                backward(out)
            del out
            synchronize(device)
            if i == 0:
                result["first_iter_s"] = timeit.default_timer() - start

        stage = "forward"
        forward_times = []
        for _ in range(steps):
            start = timeit.default_timer()
            with nvtx.range("forward"):
                attn(Q, K, V, mask)
                synchronize(device)
            forward_times.append(timeit.default_timer() - start)

        torch.cuda.reset_peak_memory_stats(device)
        out = attn(Q, K, V, mask)
        synchronize(device)
        result["mem_before_backward_bytes"] = torch.cuda.memory_allocated(device)
        if not run_backward:
            del out
            result["peak_mem_bytes"] = torch.cuda.max_memory_allocated(device)
            result.update(status="ok", forward=summarize(forward_times))
            return result
        stage = "backward"
        backward(out)
        del out
        synchronize(device)

        backward_times = []
        for _ in range(steps):
            out = attn(Q, K, V, mask)
            synchronize(device)
            start = timeit.default_timer()
            with nvtx.range("backward"):
                backward(out)
                synchronize(device)
            backward_times.append(timeit.default_timer() - start)
            del out

        result["peak_mem_bytes"] = torch.cuda.max_memory_allocated(device)
        result.update(status="ok", forward=summarize(forward_times), backward=summarize(backward_times))
    except torch.OutOfMemoryError:
        result["status"] = f"oom_{stage}"
    except OutOfResources:
        result["status"] = "out_of_shared_memory"
    finally:
        # Drop every local tensor (including a half-built graph) before the next config.
        Q = K = V = mask = grad_out = inputs = out = None  # noqa: F841
        gc.collect()
        torch.cuda.empty_cache()
    return result


def format_row(r: dict) -> str:
    head = f"{r['impl']:>13} d={r['d_model']:>4} seq={r['seq_len']:>6}"
    if r["status"] != "ok":
        mem = f" | mem before bwd {r['mem_before_backward_bytes'] / 2**20:9.1f} MiB" if "mem_before_backward_bytes" in r else ""
        return f"{head} | {r['status'].upper()}{mem}"
    bwd = f"{r['backward']['mean_ms']:9.3f} ms" if "backward" in r else f"{'n/a':>12}"
    return (
        f"{head} | fwd {r['forward']['mean_ms']:9.3f} ms | bwd {bwd}"
        f" | mem before bwd {r['mem_before_backward_bytes'] / 2**20:9.1f} MiB | peak {r['peak_mem_bytes'] / 2**20:9.1f} MiB"
    )


def main(argv: list[str] | None = None) -> list[dict]:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.causal and FORWARD_ONLY_IMPLS & set(args.impls):
        raise SystemExit("flash_pytorch / flash_triton ignore masks, so they cannot be run with --causal")
    print(f"impls={args.impls} flash_tile_size={args.flash_tile_size} batch_size={args.batch_size} causal={args.causal} warmup={args.warmup_steps} steps={args.steps} device={device}")
    results = []
    for d_model, seq_len, impl in itertools.product(sorted(args.d_models), sorted(args.seq_lens), args.impls):
        attn = make_attn(impl, args.flash_tile_size)
        run_backward = impl not in FORWARD_ONLY_IMPLS
        r = bench_config(attn, args.batch_size, seq_len, d_model, args.causal, args.warmup_steps, args.steps, device, run_backward)
        r.update(impl=impl, warmup_steps=args.warmup_steps, steps=args.steps)
        if impl in FORWARD_ONLY_IMPLS:
            r["tile_size"] = args.flash_tile_size
        print(format_row(r), flush=True)
        results.append(r)
        if args.output:
            with open(args.output, "a") as f:
                f.write(json.dumps(r) + "\n")
    return results


if __name__ == "__main__":
    main()
