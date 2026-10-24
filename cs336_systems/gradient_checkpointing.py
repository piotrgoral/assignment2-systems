"""Peak memory of activation checkpointing strategies on a stack of N Transformer blocks (handout section 3.2).

Example:
    uv run python -m cs336_systems.gradient_checkpointing --strategy uniform --block-size 1 --output results.jsonl

Like the handout's `four_blocks` example, one TransformerBlock (xl size by default, compiled with fullgraph=True) is
applied N times, so the parameters and gradients of only one block live on the GPU and the peak memory is almost all
activations. One step = forward through the N blocks + `y.sum().backward()`.

Strategies (all use `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`):
  none       no checkpointing
  uniform    split the N blocks into segments of `--block-size` blocks and checkpoint each segment (no nesting)
  sublayer   checkpoint the attention and the feed-forward sublayer of every block separately (no nesting)
  recursive  checkpoint both halves of the stack and recurse inside each half down to single blocks (nested)

`--check` instead compares the input and parameter gradients of every strategy against `none` on a small config.
"""

import argparse
import json
import statistics
import timeit
from collections.abc import Callable

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import RotaryEmbedding, TransformerBlock
from cs336_systems.benchmark import synchronize

STRATEGIES = ("none", "uniform", "sublayer", "recursive")

MiB = 1024**2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", choices=STRATEGIES, default="uniform")
    p.add_argument("--block-size", type=int, default=1, help="blocks per checkpointed segment (uniform only)")
    p.add_argument("--num-blocks", type=int, default=32)
    # xl, as in the handout's section 3.2 snippet (16 heads).
    p.add_argument("--d-model", type=int, default=2560)
    p.add_argument("--d-ff", type=int, default=10240)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--no-compile", action="store_true", help="run the block eagerly")
    p.add_argument("--warmup-steps", type=int, default=1)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", help="append the result as one JSON line to this file")
    p.add_argument("--memory-snapshot", help="dump a CUDA memory snapshot of the last measured step to this file")
    p.add_argument("--check", action="store_true", help="check gradients of all strategies on a small config")
    args = p.parse_args(argv)
    if args.strategy == "uniform" and args.num_blocks % args.block_size:
        p.error("--num-blocks must be divisible by --block-size")
    return args


def build_block(d_model: int, d_ff: int, num_heads: int, context_length: int, device: torch.device) -> TransformerBlock:
    rope = RotaryEmbedding(context_length=context_length, dim=d_model // num_heads)
    return TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, positional_encoder=rope).to(device)


def make_forward(
    block: TransformerBlock, strategy: str, num_blocks: int, block_size: int, compile: bool
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns forward(x) applying `block` num_blocks times with the given checkpointing strategy."""
    maybe_compile = (lambda f: torch.compile(f, fullgraph=True)) if compile else (lambda f: f)
    run_block = maybe_compile(block)

    def ckpt(fn, *args):
        return checkpoint(fn, *args, use_reentrant=False)

    def run_blocks(x: torch.Tensor, n: int) -> torch.Tensor:
        for _ in range(n):
            x = run_block(x)
        return x

    if strategy == "none":
        return lambda x: run_blocks(x, num_blocks)

    if strategy == "uniform":

        def forward(x: torch.Tensor) -> torch.Tensor:
            for _ in range(num_blocks // block_size):
                x = ckpt(run_blocks, x, block_size)
            return x

        return forward

    if strategy == "sublayer":
        # The two residual sublayers of TransformerBlock.forward.
        attn_sublayer = maybe_compile(lambda x: x + block.attn(block.ln1(x)))
        ffn_sublayer = maybe_compile(lambda x: x + block.ffn(block.ln2(x)))

        def forward(x: torch.Tensor) -> torch.Tensor:
            for _ in range(num_blocks):
                x = ckpt(attn_sublayer, x)
                x = ckpt(ffn_sublayer, x)
            return x

        return forward

    if strategy == "recursive":

        def recurse(x: torch.Tensor, n: int) -> torch.Tensor:
            if n == 1:
                return run_block(x)
            half = n // 2
            x = ckpt(recurse, x, half)
            return ckpt(recurse, x, n - half)

        return lambda x: recurse(x, num_blocks)

    raise ValueError(f"unknown strategy {strategy!r}")


class SavedTensorCounter:
    """Counts the bytes autograd saves for backward outside any checkpoint (i.e. the checkpoints themselves), using the
    handout's pack hook; parameters are skipped."""

    def __init__(self):
        self.bytes = 0

    def pack(self, t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.nn.Parameter):
            self.bytes += t.numel() * t.element_size()
        return t

    def hooks(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, lambda t: t)


def measure(args: argparse.Namespace, device: torch.device) -> dict:
    block = build_block(args.d_model, args.d_ff, args.num_heads, args.context_length, device)
    forward = make_forward(block, args.strategy, args.num_blocks, args.block_size, not args.no_compile)
    shape = (args.batch_size, args.context_length, args.d_model)

    def step(counter: SavedTensorCounter | None = None) -> None:
        block.zero_grad(set_to_none=False)
        x = torch.randn(shape, device=device, requires_grad=True)
        if counter is None:
            y = forward(x)
        else:
            with counter.hooks():
                y = forward(x)
        y.sum().backward()

    for _ in range(args.warmup_steps):
        step()
    synchronize(device)

    torch.cuda.empty_cache()
    base_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    if args.memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    times_ms, saved_bytes = [], []
    for _ in range(args.steps):
        counter = SavedTensorCounter()
        start = timeit.default_timer()
        step(counter)
        synchronize(device)
        times_ms.append((timeit.default_timer() - start) * 1e3)
        saved_bytes.append(counter.bytes)
    if args.memory_snapshot:
        torch.cuda.memory._dump_snapshot(args.memory_snapshot)
        torch.cuda.memory._record_memory_history(enabled=None)
    peak_bytes = torch.cuda.max_memory_allocated(device)

    return dict(
        status="ok",
        base_mib=base_bytes / MiB,
        peak_mib=peak_bytes / MiB,
        activation_peak_mib=(peak_bytes - base_bytes) / MiB,
        saved_outside_checkpoints_mib=saved_bytes[-1] / MiB,
        step_mean_ms=statistics.mean(times_ms),
        step_std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        times_ms=times_ms,
    )


def check_gradients(device: torch.device) -> None:
    """Every strategy must give the same gradients as `none` (small config, eager and compiled)."""
    torch.manual_seed(0)
    num_blocks, shape = 6, (2, 64, 128)
    block = build_block(d_model=128, d_ff=512, num_heads=4, context_length=64, device=device)
    x0 = torch.randn(shape, device=device)

    def grads(strategy: str, block_size: int, compile: bool) -> list[torch.Tensor]:
        block.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_(True)
        make_forward(block, strategy, num_blocks, block_size, compile)(x).sum().backward()
        return [x.grad] + [p.grad for p in block.parameters()]

    # Compiled kernels fuse differently (block vs. sublayers), so compare relative errors against eager `none`.
    reference = grads("none", 1, compile=False)
    for compile in (False, True):
        for strategy, block_size in [("none", 1), ("uniform", 1), ("uniform", 2), ("uniform", 3), ("sublayer", 1), ("recursive", 1)]:
            got = grads(strategy, block_size, compile)
            rel_err = max(((g - r).norm() / r.norm()).item() for g, r in zip(got, reference))
            ok = rel_err < 1e-5
            print(f"compile={compile} {strategy:>9} block_size={block_size}: {'ok' if ok else 'MISMATCH'} (max rel err {rel_err:.2e})")
            assert ok


def main(argv: list[str] | None = None) -> dict | None:
    args = parse_args(argv)
    device = torch.device("cuda")
    if args.check:
        check_gradients(device)
        return None
    torch.manual_seed(args.seed)

    try:
        result = measure(args, device)
    except torch.cuda.OutOfMemoryError:
        result = dict(status="oom", peak_mib=torch.cuda.max_memory_allocated(device) / MiB)
        torch.cuda.empty_cache()

    result = dict(
        strategy=args.strategy,
        block_size=args.block_size if args.strategy == "uniform" else None,
        num_blocks=args.num_blocks,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        context_length=args.context_length,
        batch_size=args.batch_size,
        compiled=not args.no_compile,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        **result,
    )

    label = args.strategy + (f" block_size={args.block_size}" if args.strategy == "uniform" else "")
    if result["status"] == "ok":
        print(
            f"{label}: peak {result['peak_mib']:.0f} MiB (activations {result['activation_peak_mib']:.0f} MiB, "
            f"saved outside checkpoints {result['saved_outside_checkpoints_mib']:.0f} MiB) | "
            f"step {result['step_mean_ms']:.0f} ± {result['step_std_ms']:.0f} ms"
        )
    else:
        print(f"{label}: OOM (peak allocated before failing {result['peak_mib']:.0f} MiB)")

    if args.output:
        with open(args.output, "a") as f:
            f.write(json.dumps(result) + "\n")
    return result


if __name__ == "__main__":
    main()
