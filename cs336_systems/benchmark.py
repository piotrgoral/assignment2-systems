"""End-to-end benchmarking of the cs336_basics Transformer (forward / backward / optimizer step).

Example:
    uv run python -m cs336_systems.benchmark --size small --mode train --warmup-steps 5 --steps 10 --output results.jsonl

Each step is also split into phases (forward / backward / optimizer), each followed by a synchronize,
so the backward pass and optimizer step get their own timings.
"""

import argparse
import json
import statistics
import timeit
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
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
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", help="append the result as one JSON line to this file")
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
    """Times named phases of one step; synchronizes at the end of each phase."""

    def __init__(self, device: torch.device):
        self.device = device
        self.times: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        start = timeit.default_timer()
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
) -> StepFn:
    if mode == "forward":

        def step(phase: PhaseTimer) -> None:
            with phase("forward"):
                if no_grad:
                    with torch.inference_mode():
                        cross_entropy(model(x), y)
                else:
                    cross_entropy(model(x), y)

    elif mode == "forward_backward":

        def step(phase: PhaseTimer) -> None:
            model.zero_grad(set_to_none=True)
            with phase("forward"):
                loss = cross_entropy(model(x), y)
            with phase("backward"):
                loss.backward()

    elif mode == "train":
        assert optimizer is not None, "train mode needs an optimizer"

        def step(phase: PhaseTimer) -> None:
            optimizer.zero_grad(set_to_none=True)
            with phase("forward"):
                loss = cross_entropy(model(x), y)
            with phase("backward"):
                loss.backward()
            with phase("optimizer"):
                optimizer.step()

    else:
        raise ValueError(f"unknown mode {mode!r}")
    return step


def benchmark(step_fn: StepFn, warmup_steps: int, steps: int, device: torch.device) -> list[dict[str, float]]:
    """Returns per-step wall-clock times in seconds, as {"total": ..., <phase>: ...}; synchronizes after every step."""
    for _ in range(warmup_steps):
        step_fn(PhaseTimer(device))
        synchronize(device)

    times = []
    for _ in range(steps):
        phase = PhaseTimer(device)
        start = timeit.default_timer()
        step_fn(phase)
        synchronize(device)
        times.append(dict(total=timeit.default_timer() - start, **phase.times))
    return times


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

    cfg = model_config(args)
    model = build_model(cfg, device)
    x, y = make_batch(args.batch_size, args.context_length, args.vocab_size, device)
    optimizer = AdamW(model.parameters()) if args.mode == "train" else None
    step_fn = make_step_fn(model, x, y, args.mode, optimizer, args.no_grad)

    times = benchmark(step_fn, args.warmup_steps, args.steps, device)

    n_params = sum(p.numel() for p in model.parameters())
    result = dict(
        size=size_label(args),
        **cfg,
        n_params=n_params,
        batch_size=args.batch_size,
        mode=args.mode,
        no_grad=args.no_grad,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        device=str(device),
        timings={name: summarize([t[name] for t in times]) for name in times[0]},
    )

    print(f"config: {cfg} | batch_size={args.batch_size} | params={n_params / 1e6:.1f}M | device={device}")
    print(f"mode={args.mode}{' (no_grad)' if args.no_grad else ''} warmup={args.warmup_steps} steps={args.steps}")
    for name, t in result["timings"].items():
        print(f"{name:>9}: mean {t['mean_ms']:.2f} ms | std {t['std_ms']:.2f} ms | min {t['min_ms']:.2f} ms | max {t['max_ms']:.2f} ms")

    if args.output:
        with open(args.output, "a") as f:
            f.write(json.dumps(result) + "\n")
    return result


if __name__ == "__main__":
    main()
