"""End-to-end benchmarking of the cs336_basics Transformer (forward / backward / optimizer step).

Example:
    uv run python -m cs336_systems.benchmark --size small --mode train --warmup-steps 5 --steps 10 --output results.jsonl

Each step is also split into phases (forward / backward / optimizer), each followed by a synchronize,
so the backward pass and optimizer step get their own timings.

Warmup, the measured steps and each phase are wrapped in NVTX ranges ("warmup", "measure", "step_<i>",
"forward", "backward", "optimizer"), so an nsys profile can skip warmup with
`--capture-range=nvtx --nvtx-capture=measure` (needs NSYS_NVTX_PROFILER_REGISTER_ONLY=0).

`--compile` wraps the model in torch.compile (the loss and the optimizer stay eager). Compilation happens in the
first warm-up step, whose time is reported as "first_step_s".

`--mixed-precision bf16` runs the forward pass and the loss under torch.autocast with bfloat16 (params stay fp32;
backward and the optimizer step run outside autocast).
"""

import argparse
import json
import math
import statistics
import timeit
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.optimizer import AdamW

# Table 1 of the handout.
MODEL_SIZES = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    # Disabled: don't fit for training on a 24 GB A10.
    # "xl": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    # "10B": dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

MODES = ("forward", "forward_backward", "train")

MIXED_PRECISION_DTYPES = {"none": None, "bf16": torch.bfloat16}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", choices=MODEL_SIZES, default="small", help="model size preset (Table 1)")
    p.add_argument("--d-model", type=int, help="overrides the preset")
    p.add_argument("--d-ff", type=int, help="overrides the preset")
    p.add_argument("--num-layers", type=int, help="overrides the preset")
    p.add_argument("--num-heads", type=int, help="overrides the preset")
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--rope-theta", type=float, default=10_000.0)
    p.add_argument("--mode", choices=MODES, default="train", help="what one timed step consists of")
    p.add_argument("--no-grad", action="store_true", help="run forward mode under torch.inference_mode()")
    p.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    p.add_argument(
        "--mixed-precision",
        choices=MIXED_PRECISION_DTYPES,
        default="none",
        help="run forward + loss under torch.autocast with this dtype",
    )
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", help="append the result as one JSON line to this file")
    p.add_argument(
        "--annotate-attention",
        action="store_true",
        help="swap in scaled dot-product attention with NVTX ranges around its parts",
    )
    args = p.parse_args(argv)
    if args.no_grad and args.mode != "forward":
        p.error("--no-grad only applies to --mode forward")
    return args


def size_label(args: argparse.Namespace) -> str:
    """The preset name, or "custom" if any of its dimensions were overridden."""
    preset = MODEL_SIZES[args.size]
    overridden = any(getattr(args, key) not in (None, value) for key, value in preset.items())
    return "custom" if overridden else args.size


def model_config(args: argparse.Namespace) -> dict:
    cfg = dict(MODEL_SIZES[args.size])
    for key in cfg:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    return dict(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        **cfg,
    )


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """cs336_basics.model.scaled_dot_product_attention with NVTX ranges around scores / softmax / final matmul."""
    with nvtx.range("computing attention scores"):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def build_model(cfg: dict, device: torch.device) -> BasicsTransformerLM:
    model = BasicsTransformerLM(**cfg).to(device)
    model.train()
    return model


def make_batch(batch_size: int, context_length: int, vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class PhaseTimer:
    """Times named phases of one step (each in an NVTX range); synchronizes at the end of each phase."""

    def __init__(self, device: torch.device):
        self.device = device
        self.times: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        start = timeit.default_timer()
        with nvtx.range(name):
            yield
            synchronize(self.device)
        self.times[name] = timeit.default_timer() - start


StepFn = Callable[[PhaseTimer], None]


def make_step_fn(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    optimizer: torch.optim.Optimizer | None = None,
    no_grad: bool = False,
    autocast: Callable[[], AbstractContextManager] = nullcontext,
) -> StepFn:
    """`autocast` is called to get the context manager around forward + loss (e.g. a torch.autocast factory)."""
    if mode == "forward":

        def step(phase: PhaseTimer) -> None:
            with phase("forward"):
                with torch.inference_mode() if no_grad else nullcontext(), autocast():
                    cross_entropy(model(x), y)

    elif mode == "forward_backward":

        def step(phase: PhaseTimer) -> None:
            model.zero_grad(set_to_none=True)
            with phase("forward"), autocast():
                loss = cross_entropy(model(x), y)
            with phase("backward"):
                loss.backward()

    elif mode == "train":
        assert optimizer is not None, "train mode needs an optimizer"

        def step(phase: PhaseTimer) -> None:
            optimizer.zero_grad(set_to_none=True)
            with phase("forward"), autocast():
                loss = cross_entropy(model(x), y)
            with phase("backward"):
                loss.backward()
            with phase("optimizer"):
                optimizer.step()

    else:
        raise ValueError(f"unknown mode {mode!r}")
    return step


def benchmark(
    step_fn: StepFn, warmup_steps: int, steps: int, device: torch.device
) -> tuple[list[dict[str, float]], float | None]:
    """Returns per-step wall-clock times in seconds, as {"total": ..., <phase>: ...}, and the time of the first warm-up
    step (None without warm-up); synchronizes after every step."""
    first_step_s = None
    with nvtx.range("warmup"):
        for i in range(warmup_steps):
            start = timeit.default_timer()
            step_fn(PhaseTimer(device))
            synchronize(device)
            if i == 0:
                first_step_s = timeit.default_timer() - start

    times = []
    with nvtx.range("measure"):
        for i in range(steps):
            with nvtx.range(f"step_{i}"):
                phase = PhaseTimer(device)
                start = timeit.default_timer()
                step_fn(phase)
                synchronize(device)
                times.append(dict(total=timeit.default_timer() - start, **phase.times))
    return times, first_step_s


def summarize(times_s: list[float]) -> dict:
    times_ms = [t * 1e3 for t in times_s]
    return dict(
        mean_ms=statistics.mean(times_ms),
        std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        times_ms=times_ms,
    )


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.annotate_attention:
        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    cfg = model_config(args)
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    x, y = make_batch(args.batch_size, args.context_length, args.vocab_size, device)
    optimizer = AdamW(model.parameters()) if args.mode == "train" else None
    if args.compile:
        model = torch.compile(model)
    autocast_dtype = MIXED_PRECISION_DTYPES[args.mixed_precision]
    autocast = (lambda: torch.autocast(device_type=device.type, dtype=autocast_dtype)) if autocast_dtype else nullcontext
    step_fn = make_step_fn(model, x, y, args.mode, optimizer, args.no_grad, autocast)

    times, first_step_s = benchmark(step_fn, args.warmup_steps, args.steps, device)

    result = dict(
        size=size_label(args),
        **cfg,
        n_params=n_params,
        batch_size=args.batch_size,
        mode=args.mode,
        no_grad=args.no_grad,
        compiled=args.compile,
        mixed_precision=args.mixed_precision,
        first_step_s=first_step_s,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        device=str(device),
        timings={name: summarize([t[name] for t in times]) for name in times[0]},
    )

    print(f"config: {cfg} | batch_size={args.batch_size} | params={n_params / 1e6:.1f}M | device={device}")
    print(f"mode={args.mode}{' (no_grad)' if args.no_grad else ''} compiled={args.compile} mixed_precision={args.mixed_precision} warmup={args.warmup_steps} steps={args.steps}")
    for name, t in result["timings"].items():
        print(f"{name:>9}: mean {t['mean_ms']:.2f} ms | std {t['std_ms']:.2f} ms | min {t['min_ms']:.2f} ms | max {t['max_ms']:.2f} ms")

    if args.output:
        with open(args.output, "a") as f:
            f.write(json.dumps(result) + "\n")
    return result


if __name__ == "__main__":
    main()
