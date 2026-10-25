# 4.2 flash_benchmarking (a): FlashAttention-2 vs regular PyTorch attention

**Setup.** One **NVIDIA A10 (24 GB)**, not the B200 the handout asks for (the only GPU available). Batch 1, single
head, causal masking. seq_len 2^7 … 2^16 × d_model {16, 32, 64, 128} × {bf16, fp32}. Inputs (Q, K, V, dO) are
generated before timing. Each number is the median of `triton.testing.do_bench` (25 ms warm-up, 100 ms rep, L2
flushed between runs), in ms. Each stage gets one untimed call first, so autotuning and torch.compile are not timed.

- **pytorch**: cs336_basics `scaled_dot_product_attention` with a precomputed lower-triangular bool mask (as in 4.1).
- **flash**: `FlashAttentionTriton.apply(Q, K, V, True)`. The forward is the Triton kernel, with tiles / warps / stages
  autotuned per (N, d, dtype). Key tiles past the diagonal are skipped. With fp32 inputs, both `tl.dot`s use
  `input_precision="ieee"` (true fp32, no TF32), so the fp32 rows compare like for like with the PyTorch baseline,
  which also runs true fp32 matmuls (PyTorch's default). The backward is the compiled-PyTorch
  recomputation backward (Eq. 13–19).
- **bwd** is `out.backward(dO, retain_graph=True)` on one saved output. **e2e** is `attn(Q, K, V).backward(dO)`.
  Speedup = pytorch / flash. "tiles" is the autotuned forward config Bq×Bk (num_warps / num_stages).
- OOM = out of memory in that stage. Stages are run fwd → bwd → e2e, so a backward OOM still leaves a forward time.

### bfloat16

| d | seq | pytorch fwd | pytorch bwd | pytorch e2e | flash fwd | flash bwd | flash e2e | fwd speedup | bwd speedup | e2e speedup | tiles Bq×Bk (warps/stages) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 16 | 128 | 0.039 | 0.071 | 0.377 | 0.006 | 0.045 | 0.086 | 6.3× | 1.6× | 4.4× | 32×64 (4/3) |
| 16 | 256 | 0.040 | 0.091 | 0.490 | 0.007 | 0.052 | 0.057 | 5.6× | 1.7× | 8.6× | 32×64 (4/3) |
| 16 | 512 | 0.047 | 0.090 | 0.397 | 0.009 | 0.064 | 0.071 | 5.1× | 1.4× | 5.6× | 64×64 (4/3) |
| 16 | 1024 | 0.070 | 0.155 | 0.402 | 0.014 | 0.120 | 0.131 | 4.9× | 1.3× | 3.1× | 32×64 (4/3) |
| 16 | 2048 | 0.256 | 0.589 | 0.842 | 0.024 | 0.344 | 0.365 | 10.9× | 1.7× | 2.3× | 64×64 (4/3) |
| 16 | 4096 | 0.961 | 2.13 | 3.10 | 0.043 | 1.18 | 1.23 | 22.3× | 1.8× | 2.5× | 64×64 (4/3) |
| 16 | 8192 | 3.68 | 8.11 | 11.81 | 0.104 | 4.47 | 4.60 | 35.2× | 1.8× | 2.6× | 64×64 (4/3) |
| 16 | 16384 | 14.40 | 32.07 | 46.86 | 0.308 | 17.54 | 17.92 | 46.7× | 1.8× | 2.6× | 128×128 (4/3) |
| 16 | 32768 | 57.38 | 127.98 | 186.04 | 1.12 | 69.98 | 71.40 | 51.3× | 1.8× | 2.6× | 128×128 (4/3) |
| 16 | 65536 | OOM | OOM | OOM | 3.91 | OOM | OOM | – | – | – | 128×128 (4/3) |
| 32 | 128 | 0.040 | 0.069 | 0.370 | 0.006 | 0.046 | 0.054 | 6.5× | 1.5× | 6.8× | 32×64 (4/3) |
| 32 | 256 | 0.040 | 0.073 | 0.380 | 0.008 | 0.051 | 0.057 | 4.9× | 1.4× | 6.6× | 32×64 (4/3) |
| 32 | 512 | 0.044 | 0.087 | 0.384 | 0.011 | 0.065 | 0.073 | 3.9× | 1.3× | 5.3× | 32×64 (4/3) |
| 32 | 1024 | 0.072 | 0.154 | 0.375 | 0.017 | 0.123 | 0.137 | 4.1× | 1.3× | 2.7× | 64×64 (4/3) |
| 32 | 2048 | 0.256 | 0.580 | 0.834 | 0.032 | 0.362 | 0.393 | 8.1× | 1.6× | 2.1× | 64×64 (4/3) |
| 32 | 4096 | 0.967 | 2.12 | 3.09 | 0.057 | 1.25 | 1.33 | 16.9× | 1.7× | 2.3× | 64×64 (4/3) |
| 32 | 8192 | 3.70 | 8.14 | 11.84 | 0.140 | 4.91 | 5.10 | 26.4× | 1.7× | 2.3× | 64×64 (4/3) |
| 32 | 16384 | 14.46 | 32.15 | 47.04 | 0.441 | 19.44 | 19.69 | 32.8× | 1.7× | 2.4× | 64×64 (4/3) |
| 32 | 32768 | 57.69 | 128.69 | 187.10 | 1.64 | 74.34 | 76.07 | 35.2× | 1.7× | 2.5× | 64×64 (4/3) |
| 32 | 65536 | OOM | OOM | OOM | 6.06 | OOM | OOM | – | – | – | 128×64 (8/3) |
| 64 | 128 | 0.038 | 0.068 | 0.351 | 0.007 | 0.048 | 0.053 | 5.1× | 1.4× | 6.6× | 32×64 (4/3) |
| 64 | 256 | 0.040 | 0.072 | 0.385 | 0.010 | 0.055 | 0.062 | 3.9× | 1.3× | 6.2× | 64×32 (4/3) |
| 64 | 512 | 0.045 | 0.089 | 0.365 | 0.014 | 0.072 | 0.083 | 3.1× | 1.2× | 4.4× | 64×32 (4/3) |
| 64 | 1024 | 0.076 | 0.156 | 0.414 | 0.025 | 0.149 | 0.171 | 3.1× | 1.0× | 2.4× | 64×32 (4/3) |
| 64 | 2048 | 0.266 | 0.591 | 0.861 | 0.045 | 0.441 | 0.491 | 5.9× | 1.3× | 1.8× | 64×32 (4/3) |
| 64 | 4096 | 1.02 | 2.21 | 3.25 | 0.087 | 1.52 | 1.63 | 11.8× | 1.5× | 2.0× | 64×32 (4/3) |
| 64 | 8192 | 3.76 | 8.26 | 12.04 | 0.231 | 5.91 | 6.07 | 16.2× | 1.4× | 2.0× | 64×32 (4/3) |
| 64 | 16384 | 14.87 | 32.68 | 47.83 | 0.755 | 23.36 | 24.52 | 19.7× | 1.4× | 2.0× | 64×32 (4/3) |
| 64 | 32768 | 59.57 | 130.60 | 191.50 | 2.80 | 95.80 | 99.87 | 21.2× | 1.4× | 1.9× | 64×32 (4/3) |
| 64 | 65536 | OOM | OOM | OOM | 11.56 | OOM | OOM | – | – | – | 64×32 (4/2) |
| 128 | 128 | 0.039 | 0.068 | 0.334 | 0.009 | 0.051 | 0.056 | 4.2× | 1.3× | 5.9× | 32×64 (8/3) |
| 128 | 256 | 0.040 | 0.073 | 0.385 | 0.012 | 0.067 | 0.078 | 3.2× | 1.1× | 4.9× | 32×64 (4/3) |
| 128 | 512 | 0.046 | 0.088 | 0.392 | 0.018 | 0.092 | 0.109 | 2.5× | 1.0× | 3.6× | 32×64 (4/3) |
| 128 | 1024 | 0.078 | 0.169 | 0.395 | 0.033 | 0.199 | 0.227 | 2.4× | 0.9× | 1.7× | 32×64 (4/3) |
| 128 | 2048 | 0.270 | 0.609 | 0.881 | 0.060 | 0.657 | 0.726 | 4.5× | 0.9× | 1.2× | 32×64 (4/3) |
| 128 | 4096 | 1.03 | 2.22 | 3.27 | 0.165 | 2.22 | 2.41 | 6.2× | 1.0× | 1.4× | 32×64 (4/3) |
| 128 | 8192 | 3.85 | 8.31 | 12.18 | 0.532 | 8.52 | 9.03 | 7.2× | 1.0× | 1.3× | 128×64 (8/3) |
| 128 | 16384 | 15.00 | 32.85 | 48.84 | 1.63 | 33.58 | 36.42 | 9.2× | 1.0× | 1.3× | 128×128 (8/2) |
| 128 | 32768 | 59.88 | 131.56 | 193.09 | 6.23 | 137.22 | 141.24 | 9.6× | 1.0× | 1.4× | 128×64 (8/3) |
| 128 | 65536 | OOM | OOM | OOM | 23.33 | OOM | OOM | – | – | – | 128×64 (8/2) |

### float32

| d | seq | pytorch fwd | pytorch bwd | pytorch e2e | flash fwd | flash bwd | flash e2e | fwd speedup | bwd speedup | e2e speedup | tiles Bq×Bk (warps/stages) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 16 | 128 | 0.038 | 0.072 | 0.346 | 0.009 | 0.037 | 0.044 | 4.1× | 1.9× | 7.8× | 32×64 (4/3) |
| 16 | 256 | 0.042 | 0.083 | 0.329 | 0.014 | 0.045 | 0.056 | 2.9× | 1.8× | 5.8× | 16×16 (4/3) |
| 16 | 512 | 0.053 | 0.102 | 0.370 | 0.024 | 0.056 | 0.078 | 2.3× | 1.8× | 4.8× | 32×64 (4/3) |
| 16 | 1024 | 0.116 | 0.271 | 0.383 | 0.043 | 0.110 | 0.151 | 2.7× | 2.5× | 2.5× | 32×64 (4/3) |
| 16 | 2048 | 0.458 | 1.06 | 1.51 | 0.082 | 0.337 | 0.422 | 5.6× | 3.1× | 3.6× | 32×64 (4/3) |
| 16 | 4096 | 1.74 | 3.93 | 5.67 | 0.210 | 1.19 | 1.44 | 8.3× | 3.3× | 4.0× | 64×64 (4/3) |
| 16 | 8192 | 6.66 | 15.30 | 21.94 | 0.572 | 4.47 | 5.24 | 11.6× | 3.4× | 4.2× | 64×64 (4/3) |
| 16 | 16384 | 26.38 | 60.73 | 87.01 | 1.96 | 17.71 | 19.87 | 13.5× | 3.4× | 4.4× | 128×64 (4/2) |
| 16 | 32768 | 105.54 | OOM | OOM | 6.51 | 70.86 | 79.35 | 16.2× | – | – | 128×128 (8/2) |
| 16 | 65536 | OOM | OOM | OOM | 23.87 | OOM | OOM | – | – | – | 128×128 (8/2) |
| 32 | 128 | 0.039 | 0.072 | 0.326 | 0.012 | 0.039 | 0.050 | 3.2× | 1.8× | 6.5× | 32×64 (4/3) |
| 32 | 256 | 0.042 | 0.079 | 0.413 | 0.020 | 0.043 | 0.061 | 2.1× | 1.8× | 6.7× | 32×64 (4/3) |
| 32 | 512 | 0.053 | 0.104 | 0.381 | 0.035 | 0.055 | 0.087 | 1.5× | 1.9× | 4.4× | 32×64 (4/3) |
| 32 | 1024 | 0.118 | 0.276 | 0.390 | 0.066 | 0.115 | 0.177 | 1.8× | 2.4× | 2.2× | 32×64 (4/3) |
| 32 | 2048 | 0.463 | 1.07 | 1.54 | 0.127 | 0.356 | 0.489 | 3.6× | 3.0× | 3.1× | 32×64 (4/3) |
| 32 | 4096 | 1.77 | 3.98 | 5.75 | 0.347 | 1.25 | 1.65 | 5.1× | 3.2× | 3.5× | 64×64 (4/3) |
| 32 | 8192 | 6.80 | 15.45 | 22.23 | 0.972 | 4.95 | 6.13 | 7.0× | 3.1× | 3.6× | 64×64 (4/2) |
| 32 | 16384 | 26.98 | 61.32 | 88.30 | 3.35 | 19.58 | 23.24 | 8.1× | 3.1× | 3.8× | 64×64 (4/3) |
| 32 | 32768 | 107.66 | OOM | OOM | 12.19 | 74.42 | 91.54 | 8.8× | – | – | 64×64 (4/3) |
| 32 | 65536 | OOM | OOM | OOM | 47.07 | OOM | OOM | – | – | – | 128×64 (8/2) |
| 64 | 128 | 0.040 | 0.073 | 0.351 | 0.019 | 0.040 | 0.057 | 2.1× | 1.8× | 6.1× | 32×64 (4/2) |
| 64 | 256 | 0.042 | 0.083 | 0.395 | 0.034 | 0.048 | 0.080 | 1.2× | 1.7× | 4.9× | 32×64 (4/3) |
| 64 | 512 | 0.055 | 0.108 | 0.443 | 0.062 | 0.063 | 0.123 | 0.9× | 1.7× | 3.6× | 32×64 (4/3) |
| 64 | 1024 | 0.129 | 0.302 | 0.433 | 0.120 | 0.141 | 0.259 | 1.1× | 2.1× | 1.7× | 32×64 (4/3) |
| 64 | 2048 | 0.496 | 1.12 | 1.63 | 0.235 | 0.434 | 0.689 | 2.1× | 2.6× | 2.4× | 32×64 (4/3) |
| 64 | 4096 | 1.87 | 4.15 | 6.04 | 0.658 | 1.55 | 2.31 | 2.8× | 2.7× | 2.6× | 32×64 (4/2) |
| 64 | 8192 | 7.22 | 16.16 | 23.39 | 2.18 | 6.18 | 8.39 | 3.3× | 2.6× | 2.8× | 32×64 (4/3) |
| 64 | 16384 | 28.38 | 63.41 | 91.71 | 7.88 | 24.34 | 32.70 | 3.6× | 2.6× | 2.8× | 32×64 (4/3) |
| 64 | 32768 | 114.49 | OOM | OOM | 30.27 | 111.20 | 135.70 | 3.8× | – | – | 32×64 (4/2) |
| 64 | 65536 | OOM | OOM | OOM | 122.77 | OOM | OOM | – | – | – | 32×64 (4/2) |
| 128 | 128 | 0.041 | 0.075 | 0.379 | 0.034 | 0.044 | 0.075 | 1.2× | 1.7× | 5.1× | 16×16 (4/3) |
| 128 | 256 | 0.045 | 0.123 | 0.517 | 0.061 | 0.058 | 0.117 | 0.7× | 2.1× | 4.4× | 16×16 (4/3) |
| 128 | 512 | 0.065 | 0.125 | 0.378 | 0.119 | 0.083 | 0.200 | 0.5× | 1.5× | 1.9× | 16×16 (4/3) |
| 128 | 1024 | 0.146 | 0.336 | 0.483 | 0.233 | 0.186 | 0.419 | 0.6× | 1.8× | 1.2× | 16×16 (4/3) |
| 128 | 2048 | 0.573 | 1.27 | 1.87 | 0.496 | 0.647 | 1.19 | 1.2× | 2.0× | 1.6× | 32×64 (8/2) |
| 128 | 4096 | 2.15 | 4.64 | 6.81 | 1.41 | 2.21 | 3.73 | 1.5× | 2.1× | 1.8× | 32×64 (8/2) |
| 128 | 8192 | 8.11 | 17.79 | 25.96 | 4.51 | 8.61 | 13.30 | 1.8× | 2.1× | 2.0× | 32×64 (8/2) |
| 128 | 16384 | 32.22 | 70.19 | 102.79 | 16.62 | 34.32 | 52.78 | 1.9× | 2.0× | 1.9× | 32×64 (8/2) |
| 128 | 32768 | 130.46 | OOM | OOM | 65.95 | 140.20 | 213.59 | 2.0× | – | – | 32×64 (8/2) |
| 128 | 65536 | OOM | OOM | OOM | 267.50 | OOM | OOM | – | – | – | 32×64 (8/2) |

**Takeaways.**
- **The Triton forward wins almost everywhere.** In bf16 it is 2.4–6.5× faster at N ≤ 1024, where PyTorch is
  bound by launches and allocations, and 9–51× faster at N ≥ 16k. In fp32 it is 1.9–16× faster at N ≥ 16k. The gap is
  largest at small d, where eager attention is pure N² memory traffic (scores, mask, softmax) and the fused kernel
  never writes S or P to HBM (bf16 at N ≥ 16k: 47–51× at d = 16, 9–10× at d = 128). At d = 128 the kernel becomes
  compute-bound: bf16 at N = 65536 takes 23.3 ms for 2·N²·d ≈ 1.1 TFLOP (causal half), which is ~47 TFLOP/s,
  ~38% of the A10's 125 TFLOP/s bf16 tensor-core peak.
- **fp32 is where the kernel is weakest.** With true fp32 `tl.dot` there are no tensor cores, so the same shape
  takes 267.5 ms (~4.1 TFLOP/s, ~13% of the A10's 31 TFLOP/s fp32 peak), which is 11.5× slower than bf16. The
  fp32 forward advantage over eager shrinks with d (at N ≥ 16k: 13.5–16× at d = 16, 3.6–3.8× at d = 64,
  1.9–2.0× at d = 128). At d = 128, N = 256–1024 (and d = 64, N = 512) it is even slower than eager (0.5–0.9×),
  most likely because cuBLAS's fp32 GEMM uses the FMA units better than Triton's IEEE dot does. A TF32 kernel (Triton's default) is ~4× faster here (median
  3.7× over all fp32 shapes), but comparing it with a true-fp32 baseline would overstate the speedup.
- **The backward dominates, and it is barely faster than eager.** The compiled backward recomputes P with full
  N × N fp32 matrices, so it is still O(N²) in memory traffic. At N ≥ 4096, flash bwd is ≥ 92% of flash e2e in
  bf16 and ≥ 59% in fp32. Against eager (N ≥ 1024) it is 0.9–1.8× faster in bf16 and 1.8–3.4× in fp32. In bf16 at
  d = 128 it is no faster (0.9–1.0×), because it upcasts everything to fp32 (bf16 and fp32 flash bwd times are
  almost the same) while eager runs bf16 matmuls. So e2e speedups at N = 16384 are only 1.3–2.6× (bf16) and
  1.9–4.4× (fp32). A tiled Triton backward (Algorithm 2) is the obvious next step.
- **Memory.** Eager PyTorch OOMs in the forward at N = 65536 (the N² bool mask plus N² scores and probabilities) and
  in fp32 backward from N = 32768. The flash forward runs at every size, because it only stores O(N·d) (Q, K, V, O, L).
  Only its compiled backward OOMs at N = 65536 (one fp32 N × N matrix is 16 GiB).
- **Small sequences are overhead.** Below ~1024 tokens the bf16 flash forward takes 6–33 µs (fp32: 9–233 µs). Eager
  e2e is ~0.35 ms regardless of size, which is CPU launch/autograd overhead: do_bench sees it because the GPU waits
  for the CPU.
- **Causal skipping.** Stopping the key loop at the diagonal makes the causal forward faster than the full
  non-causal loop by 1.41–1.43× (bf16, d ∈ {16, 128}) and 1.41 / 1.75× (fp32, d = 16 / 128), at N = 16384. This is
  less than 2× because query tiles near the start finish early, which leaves SMs idle at the end of the launch. The
  compute-bound fp32 d = 128 case gets closest to 2×.
- **Tiles.** Autotuning picks 32×64 for short sequences (more programs to fill the 72 SMs). In bf16 it picks
  64×64–128×128 at d = 16 and long N, 64×32 at d = 64, and 128×64–128 with 8 warps at d = 128. In fp32 it picks
  smaller tiles at large d: 32×64 at d = 64, 32×64 with 8 warps at d = 128, and 16×16 at d = 128, N ≤ 1024.
  Twice the bytes per element limits the tile and stage count, and without tensor cores the bigger tiles stop
  paying off (see the fixed-tile sweep in 4.2.3).

Raw data: `results/data/4_2_flash_benchmarking.jsonl` (one row per impl × dtype × d × N, with a `tile_config` field
for flash). Log: `results/data/4_2_flash_benchmarking.log`. Commands: `results/notes/4_2_flash_benchmarking.md`.
