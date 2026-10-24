# 2.1.3 benchmarking_script (b)

**Answer.** On an NVIDIA A10 (fp32, batch 4, context 512, 5 warm-up and 10 timed steps), one forward pass takes 73 ms (small), 233 ms (medium) and 499 ms (large). The backward pass takes about 2× as long: 154 ms, 441 ms and 953 ms. Variability is small: every standard deviation is under 1% of its mean.

| size   | params | forward (ms) | backward (ms) | optimizer (ms) | total step (ms) |
|--------|-------:|-------------:|--------------:|---------------:|----------------:|
| small  | 128.6M |  73.07 ± 0.37 |  154.27 ± 1.09 |  38.72 ± 0.04 |   266.96 ± 1.32 |
| medium | 423.2M | 232.85 ± 1.28 |  441.08 ± 1.52 | 124.31 ± 0.09 |   800.19 ± 1.65 |
| large  | 969.4M | 498.96 ± 2.32 |  953.28 ± 2.92 |            — | 1456.15 ± 5.05 † |

Values are mean ± standard deviation over 10 steps. Each phase is followed by `torch.cuda.synchronize()`. The total step is timed separately around the whole step. Small and medium were run with `--mode train`.

† Large was run with `--mode forward_backward`, so its total has no optimizer step. A full training step does not fit in the A10's 22 GB: the fp32 parameters, gradients and two AdamW states alone take about 16 B/param ≈ 15.5 GB, before activations. xl and 10B do not fit at all.

Raw data: `results/data/2_1_benchmarking_script_b.jsonl` (log: `results/data/2_1_benchmarking_script_b.log`).
