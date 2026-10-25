"""Benchmark all-reduce in the single-node multi-process setup (distributed_communication_single_node).

Examples:
    uv run python -m cs336_systems.distributed_communication --world-sizes 2 4 6 --output results/data/foo.jsonl
    uv run python -m cs336_systems.distributed_communication --backend gloo --world-sizes 2 4 --sizes-mb 1 10
    uv run python -m cs336_systems.distributed_communication --plot results/data/foo.jsonl results/writeups/foo.png

For each world size, `mp.spawn` starts a fresh process group (one process per GPU under NCCL, via
`torch.cuda.set_device(rank)`). For each size, every rank holds a float32 tensor of size_mb MiB (2**20 bytes). Before
timing, a correctness check runs: rank r fills the tensor with r + 1 and the all-reduced sum must equal ws(ws + 1)/2.
Then come `--warmup` untimed all-reduces, a barrier, and `--iters` timed ones. Each timed iteration is
`perf_counter` around `dist.all_reduce` + `torch.cuda.synchronize()`, so it measures completion rather than enqueue.
Per-rank times are collected with `dist.all_gather_object`, and each iteration's time is the max across ranks (the
slowest rank sets the collective's latency). The statistics are taken over those per-iteration maxima.
Bandwidths follow the nccl-tests convention: algbw = bytes / t, and busbw = algbw * 2(n - 1)/n, the traffic per link
of a ring all-reduce. busbw can be compared across world sizes.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    p.add_argument("--world-sizes", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--sizes-mb", type=int, nargs="+", default=[1, 10, 100, 1000])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--port", type=int, default=29500)
    p.add_argument("--output", help="append one JSON line per config to this file")
    p.add_argument("--plot", nargs=2, metavar=("IN_JSONL", "OUT_PNG"), help="plot an existing JSONL and exit")
    return p.parse_args(argv)


def setup(rank: int, world_size: int, backend: str, port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group(backend, rank=rank, world_size=world_size, device_id=device)
        return device
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_size(size_mb: int, rank: int, world_size: int, device: torch.device, args: argparse.Namespace) -> dict | None:
    numbytes = size_mb * 2**20
    data = torch.full((numbytes // 4,), float(rank + 1), dtype=torch.float32, device=device)
    dist.all_reduce(data)
    expected = world_size * (world_size + 1) / 2
    assert torch.all(data == expected), f"rank {rank}: all-reduce sum wrong for {size_mb} MiB"

    data.normal_()
    for _ in range(args.warmup):
        dist.all_reduce(data)
    sync(device)
    dist.barrier()

    times_ms = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        dist.all_reduce(data)
        sync(device)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    del data

    gathered: list[list[float]] = [None] * world_size  # type: ignore[list-item]
    dist.all_gather_object(gathered, times_ms)
    if rank != 0:
        return None
    per_iter = [max(ts) for ts in zip(*gathered)]
    median_ms = statistics.median(per_iter)
    algbw = numbytes / (median_ms / 1e3) / 1e9
    return dict(
        backend=args.backend,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        world_size=world_size,
        size_mb=size_mb,
        bytes=numbytes,
        warmup=args.warmup,
        iters=args.iters,
        mean_ms=statistics.mean(per_iter),
        median_ms=median_ms,
        std_ms=statistics.stdev(per_iter) if len(per_iter) > 1 else 0.0,
        per_rank_mean_ms=[statistics.mean(ts) for ts in gathered],
        algbw_GBps=algbw,
        busbw_GBps=algbw * 2 * (world_size - 1) / world_size,
    )


def format_row(r: dict) -> str:
    return (
        f"{r['backend']} ws={r['world_size']} size={r['size_mb']:>4} MiB | median {r['median_ms']:9.3f} ms "
        f"| mean {r['mean_ms']:9.3f} ± {r['std_ms']:7.3f} ms | algbw {r['algbw_GBps']:7.2f} GB/s "
        f"| busbw {r['busbw_GBps']:7.2f} GB/s"
    )


def worker(rank: int, world_size: int, args: argparse.Namespace, port: int) -> None:
    device = setup(rank, world_size, args.backend, port)
    try:
        for size_mb in sorted(args.sizes_mb):
            r = bench_size(size_mb, rank, world_size, device, args)
            if r is not None:
                print(format_row(r), flush=True)
                if args.output:
                    with open(args.output, "a") as f:
                        f.write(json.dumps(r) + "\n")
    finally:
        dist.destroy_process_group()


def plot(in_path: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(in_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    fig, (ax_t, ax_bw) = plt.subplots(1, 2, figsize=(11, 4))
    for ws in sorted({r["world_size"] for r in rows}):
        rs = sorted((r for r in rows if r["world_size"] == ws), key=lambda r: r["size_mb"])
        sizes = [r["size_mb"] for r in rs]
        ax_t.plot(sizes, [r["median_ms"] for r in rs], marker="o", label=f"{ws} procs")
        ax_bw.plot(sizes, [r["busbw_GBps"] for r in rs], marker="o", label=f"{ws} procs")
    for ax, ylabel in ((ax_t, "median all-reduce time (ms)"), (ax_bw, "bus bandwidth (GB/s)")):
        ax.set_xscale("log")
        ax.set_xlabel("tensor size (MiB, float32)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    ax_t.set_yscale("log")
    devices = sorted({r["device"] for r in rows})
    fig.suptitle(f"all-reduce, {rows[0]['backend']} on {', '.join(devices)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.plot:
        plot(*args.plot)
        return
    n_gpus = torch.cuda.device_count()
    print(f"backend={args.backend} gpus={n_gpus} world_sizes={args.world_sizes} sizes_mb={args.sizes_mb}")
    for i, world_size in enumerate(args.world_sizes):
        if args.backend == "nccl" and world_size > n_gpus:
            print(f"skipping world_size={world_size}: only {n_gpus} GPUs visible")
            continue
        mp.spawn(fn=worker, args=(world_size, args, args.port + i), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
