# 2.1.3 benchmarking_script (c)

**Answer.** With no warm-up, the first timed step is about 200–250 ms slower than the others for every model size (small: 519 ms vs. ~267 ms). That inflates the 10-step mean by 9% (small), 3% (medium) and 1.5% (large), and the standard deviation by ~25–90× (small: 292 ± 80 ms vs. 274 ± 1 ms with 5 warm-ups). The cause is one-time costs paid by whichever step runs first: creating the CUDA context and loading kernels, initializing cuBLAS handles and workspaces, and the caching allocator's first `cudaMalloc` calls. Because these costs don't depend on model size, the extra time is roughly constant, and most of it falls in the first forward pass (271 ms vs. 72 ms for small). With 1 or 2 warm-up steps the outlier is gone and the means fall within ~1–2% of the 5-warm-up results. They still differ slightly, and 1 warm-up has a somewhat higher std (e.g. large 6.5 ms vs. 3.0 ms). The likely reason is that the GPU hasn't reached a steady state yet: clocks are still boosting or throttling as the card heats up (means drift up by ~1% from w=1 to w=5 across the back-to-back runs), and the allocator and cuBLAS heuristics are still settling during the first iterations.

Total step time; small and medium use `--mode train`, large uses `--mode forward_backward`; 10 timed steps per run, each run in a fresh process:

| size   | warm-up | mean ± std (ms) | 1st step (ms) | mean of steps 2–10 (ms) | max / min |
|--------|--------:|----------------:|--------------:|------------------------:|----------:|
| small  | 0 |  292.2 ± 79.6 |  518.6 |  267.1 | 1.96 |
| small  | 1 |  271.0 ± 2.4  |  268.9 |  271.2 | 1.03 |
| small  | 2 |  272.3 ± 1.3  |  271.2 |  272.4 | 1.02 |
| small  | 5 |  273.6 ± 0.9  |  272.9 |  273.7 | 1.01 |
| medium | 0 |  820.8 ± 75.3 | 1034.8 |  797.0 | 1.31 |
| medium | 1 |  801.9 ± 2.7  |  799.7 |  802.1 | 1.01 |
| medium | 2 |  807.9 ± 1.4  |  810.5 |  807.6 | 1.01 |
| medium | 5 |  811.1 ± 2.3  |  809.7 |  811.2 | 1.01 |
| large  | 0 | 1478.4 ± 71.6 | 1681.3 | 1455.9 | 1.17 |
| large  | 1 | 1458.4 ± 6.5  | 1443.6 | 1460.0 | 1.02 |
| large  | 2 | 1474.1 ± 2.9  | 1472.0 | 1474.4 | 1.01 |
| large  | 5 | 1476.4 ± 3.0  | 1481.1 | 1475.8 | 1.01 |

How the first step's extra time with w=0 splits across phases (vs. the mean of later steps): forward +198 / +188 / +180 ms and backward +54 / +50 / +47 ms for small / medium / large. The optimizer step shows no first-step penalty.

Raw data: `results/data/2_1_benchmarking_script_c.jsonl` (log: `results/data/2_1_benchmarking_script_c.log`).
