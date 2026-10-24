# 2.1.4 nsys_profile (a)

**Answer.** In the profiles, one forward pass takes from 32 ms (small, ctx 256) to 286 ms (medium, ctx 1024, batch 2); small at ctx 512 takes 73.5 ms in a training step. This matches the Python `timeit` measurements of the same configurations to within 3% in every case (and to within ~1% for the training runs), so profiling with `--trace=cuda,nvtx` adds very little overhead.

nsys forward = the `forward` NVTX range (avg ± std over the 3 profiled steps, from `nvtx_sum`); timeit = the same phase measured by `benchmark.py` without nsys (mean ± std over 10 steps). Both use 5 warm-up steps, and in both the range/timer ends after `torch.cuda.synchronize()`, so they measure the same wall time.

| size | ctx | bs | mode | nsys forward (ms) | timeit forward (ms) | Δ | nsys backward | timeit backward | nsys optimizer | timeit optimizer |
|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|
| small | 256 | 4 | forward | 33.5 ± 1.7 | 34.5 ± 0.1 | -2.9% | — | — | — | — |
| small | 256 | 4 | train | 32.0 ± 0.1 | 31.6 ± 0.1 | +1.3% | 70.7 | 70.2 | 39.4 | 38.6 |
| small | 512 | 4 | forward | 75.8 ± 3.6 | 77.8 ± 1.5 | -2.5% | — | — | — | — |
| small | 512 | 4 | train | 73.5 ± 1.4 | 73.8 ± 0.6 | -0.4% | 156.2 | 155.1 | 39.5 | 38.8 |
| small | 1024 | 4 | forward | 187.1 ± 2.6 | 189.1 ± 1.3 | -1.1% | — | — | — | — |
| small | 1024 | 4 | train | 185.4 ± 0.9 | 186.5 ± 0.4 | -0.6% | 384.9 | 384.8 | 39.5 | 38.7 |
| medium | 256 | 4 | forward | 106.0 ± 3.2 | 108.4 ± 1.9 | -2.2% | — | — | — | — |
| medium | 256 | 4 | train | 102.6 ± 2.0 | 102.5 ± 0.8 | +0.1% | 208.5 | 208.7 | 126.1 | 124.5 |
| medium | 512 | 4 | forward | 235.1 ± 4.7 | 239.6 ± 2.4 | -1.9% | — | — | — | — |
| medium | 512 | 4 | train | 234.4 ± 3.6 | 234.2 ± 2.3 | +0.1% | 445.8 | 442.2 | 126.2 | 124.5 |
| medium | 1024 | 2 | forward | 286.4 ± 3.0 | 288.2 ± 2.1 | -0.6% | — | — | — | — |
| medium | 1024 | 2 | train | 284.1 ± 1.3 | 281.4 ± 0.4 | +1.0% | 563.1 | 561.9 | 126.6 | 124.9 |

The ctx 512 timeit numbers also agree with the earlier `2_1_3_benchmarking_script_b` run (small 73.8 vs 73.1 ms, medium 234.2 vs 232.9 ms). The GPU is busy for almost the whole range: in small ctx 512, the forward pass's kernels add up to 72.2 ms of the 73.5 ms wall time.

Setup: NVIDIA A10, fp32, nsys 2026.3.2, `--trace=cuda,nvtx`, capture range = `measure` (5 warm-up steps skipped, 3 steps recorded). `forward` mode runs the forward pass + loss with autograd enabled (no `--no-grad`); `train` = forward + backward + AdamW step. medium ctx 1024 uses batch 2 (batch 4 OOMs in training).

Raw data: `results/data/2_1_4_nsys_profile/*_nvtx_sum.csv`, timeit: `results/data/2_1_4_nsys_profile_timeit.jsonl` (log: `results/data/2_1_4_nsys_profile_timeit.log`).
