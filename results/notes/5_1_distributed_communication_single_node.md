# 5.1 distributed_communication_single_node — commands

## Script

`cs336_systems/distributed_communication.py`. For each world size, `mp.spawn` starts one process per GPU
(`torch.cuda.set_device(rank)`, NCCL). For each size, every rank holds a float32 tensor of 1 / 10 / 100 / 1000 MiB
(MiB = 2^20 bytes).

- Correctness check first: rank r fills the tensor with r + 1, and after `all_reduce` every element must equal
  ws(ws+1)/2.
- 5 untimed warm-up all-reduces, then `torch.cuda.synchronize()` + `dist.barrier()`.
- 20 timed iterations. Each one is `perf_counter` around `dist.all_reduce` + `torch.cuda.synchronize()`, because
  `async_op=False` only waits until the op is enqueued.
- Per-rank times are collected with `dist.all_gather_object`. Each iteration's time is the max across ranks (the
  slowest rank finishes the collective), and the median is taken over those.
- algbw = bytes / t. busbw = algbw · 2(n−1)/n, the nccl-tests convention: the per-link traffic of a ring all-reduce,
  which can be compared across world sizes.

Local smoke test (Mac, gloo/CPU):

```bash
uv run python -m cs336_systems.distributed_communication --backend gloo --world-sizes 2 4 --sizes-mb 1 10
```

## Runs (Modal, NVIDIA A10)

Each world size runs in its own container and writes to its own file, because each container starts from the local
repo and the sync would overwrite a shared file:

```bash
out=results/data/5_1_distributed_communication_single_node
for n in 2 4; do
  modal run modal_runner.py --gpu A10:$n --command \
    "python -m cs336_systems.distributed_communication --world-sizes $n --output ${out}_ws$n.jsonl"
done
cat ${out}_ws2.jsonl ${out}_ws4.jsonl > ${out}.jsonl
uv run python -m cs336_systems.distributed_communication --plot ${out}.jsonl results/writeups/5_1_distributed_communication_single_node.png
```

World size 6 was not run. `--gpu A10:6` is rejected by Modal with
`InvalidError: Function GPU request out of bounds. Must be between 1 and 4 for the given instance type.`
Running all of 2/4/6 on one L4:8 or H100:8 box would cover it.

Caveat: the 2- and 4-GPU runs were separate containers, so they may have run on different hosts with different PCIe
topologies.
