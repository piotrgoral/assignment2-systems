# 4.2 torch_compile (a)

`cs336_basics.model.scaled_dot_product_attention` in eager mode vs `torch.compile`d, fp32, batch 8, single head, no
mask, on an A10 (22.5 GiB usable). Mean of 100 timed passes after 10 warm-up passes (the first warm-up pass of the
compiled version includes compilation, 0.2–1.9 s per config). Times are in ms. "mem before bwd" is
`torch.cuda.memory_allocated()` after the forward pass, the same for both versions to within 0.2 MiB. Peak is over the
whole benchmark (forward and backward).

| d_model | seq_len | fwd eager | fwd compiled | fwd speedup | bwd eager | bwd compiled | bwd speedup | mem before bwd (MiB) | peak eager / compiled (MiB) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 256 | 0.22 | 0.19 | 1.17× | 0.75 | 0.36 | 2.07× | 20.9 | 29 / 25 |
| 16 | 1024 | 1.00 | 0.45 | 2.23× | 2.48 | 1.23 | 2.01× | 82.8 | 211 / 147 |
| 16 | 4096 | 14.67 | 6.41 | 2.29× | 37.22 | 17.59 | 2.12× | 1050.6 | 3101 / 2077 |
| 16 | 8192 | 58.01 | 25.50 | 2.27× | 147.48 | 70.60 | 2.09× | 4133.0 | 12329 / 8233 |
| 16 | 16384 | oom_forward | oom_backward | – | – | – | – | – | – |
| 32 | 256 | 0.22 | 0.19 | 1.14× | 0.51 | 0.32 | 1.58× | 21.5 | 30 / 26 |
| 32 | 1024 | 1.03 | 0.47 | 2.19× | 2.54 | 1.28 | 1.98× | 85.3 | 214 / 150 |
| 32 | 4096 | 15.12 | 6.90 | 2.19× | 37.97 | 18.32 | 2.07× | 1060.6 | 3113 / 2089 |
| 32 | 8192 | 59.76 | 27.27 | 2.19× | 149.93 | 72.92 | 2.06× | 4153.0 | 12353 / 8257 |
| 32 | 16384 | oom_forward | oom_backward | – | – | – | – | – | – |
| 64 | 256 | 0.22 | 0.19 | 1.20× | 0.48 | 0.35 | 1.39× | 22.8 | 31 / 27 |
| 64 | 1024 | 1.07 | 0.53 | 2.02× | 2.59 | 1.34 | 1.93× | 90.3 | 220 / 156 |
| 64 | 4096 | 15.45 | 7.55 | 2.04× | 38.44 | 19.14 | 2.01× | 1080.6 | 3137 / 2113 |
| 64 | 8192 | 61.27 | 30.05 | 2.04× | 152.63 | 77.06 | 1.98× | 4193.0 | 12401 / 8305 |
| 64 | 16384 | oom_forward | oom_backward | – | – | – | – | – | – |
| 128 | 256 | 0.22 | 0.19 | 1.14× | 0.48 | 0.32 | 1.51× | 25.3 | 34 / 30 |
| 128 | 1024 | 1.16 | 0.61 | 1.89× | 2.71 | 1.48 | 1.83× | 100.3 | 232 / 168 |
| 128 | 4096 | 16.95 | 9.11 | 1.86× | 40.26 | 21.21 | 1.90× | 1120.6 | 3185 / 2161 |
| 128 | 8192 | 67.42 | 38.43 | 1.75× | 160.82 | 86.55 | 1.86× | 4273.0 | 12497 / 8401 |
| 128 | 16384 | oom_forward | oom_backward | – | – | – | – | – | – |

**Answer.** Compiling gives about a 2× speedup in both passes for every configuration from seq_len 1024 up. Forward
speeds up 2.0–2.3× for d_model ≤ 64 and 1.75–1.9× for d_model = 128. Backward speeds up 1.8–2.1×. At seq_len 256
the forward pass barely speeds up (≈ 1.15×), because it only takes ≈ 0.2 ms and is dominated by kernel-launch and
Python overhead. Inductor does not replace the two matmuls (they still go to cuBLAS). It fuses the elementwise work
around them: the scaling by 1/√d, and the softmax's max, subtract, exp, sum and divide become one or two Triton
kernels. Eager runs each of these as a separate kernel that reads and writes a full B·N² tensor in HBM. That
elementwise traffic is what dominates attention at these head sizes, so fusing it roughly halves the time. The gain
shrinks a little at d_model = 128 because there the matmuls, which compiling does not speed up, are a larger share of
the time.

Compiling does not fix the memory problem. The memory saved for backward is unchanged, still two N × N fp32
tensors (4.1 GiB at N = 8192). Fusion only removes the transient intermediates, so peak memory falls by about a third
(12.3 → 8.2 GiB at N = 8192), and the compiled forward pass peaks at just those two tensors. At N = 16384 that is
16 GiB. The compiled forward pass now fits, while eager OOMs already in the forward pass (it holds four N²
tensors at once, see 4.1.1). The compiled backward still OOMs because it needs more N²-sized gradients on top of that.
Time and memory still grow as N², so long sequences need a kernel that never materializes the score matrix
(FlashAttention-2, next section).

For seq_len ≥ 1024 the eager rows match the 4.1.1 run to within ≈ 1%. At seq_len 256 they vary by up to 60%
(d = 16 backward: 0.75 ms here vs 0.46 ms in 4.1.1; it is the first config of the process). At ≈ 0.2–0.7 ms these
passes are dominated by launch overhead and are noisy.

Raw data: `results/data/4_2_torch_compile_a.jsonl` (log: `results/data/4_2_torch_compile_a.log`); commands in
`results/notes/4_2_torch_compile.md`.
