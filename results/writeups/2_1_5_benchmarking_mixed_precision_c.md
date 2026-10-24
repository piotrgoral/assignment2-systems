# 2.1.5 benchmarking_mixed_precision (c)

**Answer.** BF16 autocast speeds up forward + backward by 1.97× for small (221.9 → 112.4 ms), 2.16× for medium
(650.3 → 301.0 ms) and 2.28× for large (1394.8 → 610.7 ms), with the forward pass gaining slightly more than the
backward pass (2.03–2.44× vs. 1.96–2.23×). The speedup grows with model size: the bigger the model, the more of the step
is large matmuls, which BF16 tensor cores run much faster than FP32. The parts that don't speed up as much, like the
casts autocast adds, the norms and softmax kept in FP32, and other memory-bound elementwise kernels, take up a smaller
share as models get bigger. The gain is well below the ratio of peak FLOPs, because a lot of the step time isn't spent
in matmuls.

Mean ± std over 10 timed steps (5 warm-up), `--mode forward_backward`, batch 4, context 512, A10 24 GB:

| size | precision | forward (ms) | backward (ms) | total (ms) | speedup fwd / bwd / total |
|------|-----------|-------------:|--------------:|-----------:|--------------------------:|
| small | fp32 | 73.0 ± 1.5 | 147.4 ± 1.2 | 221.9 ± 2.0 | — |
| small | bf16 autocast | 36.0 ± 0.3 | 75.1 ± 0.4 | 112.4 ± 0.7 | 2.03× / 1.96× / 1.97× |
| medium | fp32 | 226.1 ± 0.7 | 421.5 ± 1.7 | 650.3 ± 1.3 | — |
| medium | bf16 autocast | 96.0 ± 0.4 | 202.0 ± 0.5 | 301.0 ± 0.9 | 2.36× / 2.09× / 2.16× |
| large | fp32 | 477.6 ± 1.7 | 913.3 ± 3.0 | 1394.8 ± 4.4 | — |
| large | bf16 autocast | 196.1 ± 0.6 | 410.2 ± 0.6 | 610.7 ± 1.0 | 2.44× / 2.23× / 2.28× |

Raw data: `results/data/2_1_5_benchmarking_mixed_precision_c.jsonl` (log: `results/data/2_1_5_benchmarking_mixed_precision_c.log`).
