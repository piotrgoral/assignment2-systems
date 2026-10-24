# 4.2 flash_benchmarking (a): FlashAttention-2 vs regular PyTorch attention

**Setup.** One **NVIDIA A10 (24 GB)**, not the B200 the handout asks for (the only GPU available). Batch 1, single
head, causal masking. seq_len 2^7 … 2^16 × d_model {16, 32, 64, 128} × {bf16, fp32}. Inputs (Q, K, V, dO) are
generated before timing. Each number is the median of `triton.testing.do_bench` (25 ms warm-up, 100 ms rep, L2
flushed between runs), in ms. Each stage gets one untimed call first, so autotuning and torch.compile are not timed.

- **pytorch**: cs336_basics `scaled_dot_product_attention` with a precomputed lower-triangular bool mask (as in 4.1).
- **flash**: `FlashAttentionTriton.apply(Q, K, V, True)`. The forward is the Triton kernel, with tiles / warps / stages
  autotuned per (N, d, dtype). Key tiles past the diagonal are skipped. The backward is the compiled-PyTorch
  recomputation backward (Eq. 13–19).
- **bwd** is `out.backward(dO, retain_graph=True)` on one saved output. **e2e** is `attn(Q, K, V).backward(dO)`.
  Speedup = pytorch / flash. "tiles" is the autotuned forward config Bq×Bk (num_warps / num_stages).
- OOM = out of memory in that stage. Stages are run fwd → bwd → e2e, so a backward OOM still leaves a forward time.

### bfloat16

| d | seq | pytorch fwd | pytorch bwd | pytorch e2e | flash fwd | flash bwd | flash e2e | fwd speedup | bwd speedup | e2e speedup | tiles Bq×Bk (warps/stages) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 16 | 128 | 0.039 | 0.071 | 0.359 | 0.006 | 0.045 | 0.048 | 6.3× | 1.6× | 7.5× | 32×64 (4/3) |
| 16 | 256 | 0.040 | 0.074 | 0.375 | 0.006 | 0.052 | 0.069 | 6.3× | 1.4× | 5.5× | 32×64 (4/3) |
| 16 | 512 | 0.047 | 0.090 | 0.383 | 0.009 | 0.065 | 0.075 | 5.1× | 1.4× | 5.1× | 32×64 (4/3) |
| 16 | 1024 | 0.070 | 0.156 | 0.391 | 0.014 | 0.120 | 0.130 | 4.9× | 1.3× | 3.0× | 32×64 (4/3) |
| 16 | 2048 | 0.257 | 0.588 | 0.843 | 0.024 | 0.343 | 0.366 | 10.9× | 1.7× | 2.3× | 32×64 (4/3) |
| 16 | 4096 | 0.962 | 2.13 | 3.09 | 0.050 | 1.18 | 1.24 | 19.2× | 1.8× | 2.5× | 128×128 (4/3) |
| 16 | 8192 | 3.68 | 8.12 | 11.80 | 0.098 | 4.47 | 4.60 | 37.5× | 1.8× | 2.6× | 128×128 (4/3) |
| 16 | 16384 | 14.39 | 32.05 | 46.88 | 0.285 | 17.62 | 17.90 | 50.6× | 1.8× | 2.6× | 128×128 (4/3) |
| 16 | 32768 | 57.20 | 127.73 | 186.16 | 1.05 | 70.74 | 71.81 | 54.3× | 1.8× | 2.6× | 128×128 (4/3) |
| 16 | 65536 | OOM | OOM | OOM | 3.67 | OOM | OOM | – | – | – | 128×128 (4/3) |
| 32 | 128 | 0.039 | 0.069 | 0.363 | 0.006 | 0.047 | 0.071 | 6.3× | 1.5× | 5.1× | 64×64 (4/3) |
| 32 | 256 | 0.040 | 0.073 | 0.499 | 0.007 | 0.051 | 0.201 | 5.6× | 1.4× | 2.5× | 32×64 (4/3) |
| 32 | 512 | 0.045 | 0.199 | 0.582 | 0.010 | 0.065 | 0.124 | 4.4× | 3.1× | 4.7× | 32×64 (4/3) |
| 32 | 1024 | 0.072 | 0.154 | 0.459 | 0.016 | 0.124 | 0.136 | 4.4× | 1.2× | 3.4× | 64×64 (4/3) |
| 32 | 2048 | 0.257 | 0.579 | 0.833 | 0.029 | 0.364 | 0.389 | 9.0× | 1.6× | 2.1× | 64×64 (4/3) |
| 32 | 4096 | 0.966 | 2.12 | 3.09 | 0.052 | 1.25 | 1.32 | 18.5× | 1.7× | 2.3× | 64×64 (4/3) |
| 32 | 8192 | 3.71 | 8.13 | 11.86 | 0.132 | 4.94 | 5.08 | 28.1× | 1.6× | 2.3× | 64×64 (4/3) |
| 32 | 16384 | 14.47 | 32.18 | 47.04 | 0.418 | 19.27 | 19.67 | 34.6× | 1.7× | 2.4× | 64×64 (4/3) |
| 32 | 32768 | 57.66 | 128.55 | 186.72 | 1.57 | 74.19 | 75.66 | 36.8× | 1.7× | 2.5× | 64×64 (4/3) |
| 32 | 65536 | OOM | OOM | OOM | 5.77 | OOM | OOM | – | – | – | 128×64 (8/3) |
| 64 | 128 | 0.038 | 0.068 | 0.358 | 0.007 | 0.048 | 0.074 | 5.3× | 1.4× | 4.8× | 32×64 (4/3) |
| 64 | 256 | 0.040 | 0.072 | 0.388 | 0.010 | 0.054 | 0.089 | 4.0× | 1.3× | 4.4× | 32×64 (4/3) |
| 64 | 512 | 0.045 | 0.089 | 0.442 | 0.014 | 0.072 | 0.089 | 3.1× | 1.2× | 5.0× | 32×64 (4/3) |
| 64 | 1024 | 0.077 | 0.155 | 0.421 | 0.025 | 0.150 | 0.171 | 3.1× | 1.0× | 2.5× | 32×64 (4/3) |
| 64 | 2048 | 0.266 | 0.591 | 0.859 | 0.045 | 0.441 | 0.489 | 5.9× | 1.3× | 1.8× | 32×64 (4/3) |
| 64 | 4096 | 1.02 | 2.20 | 3.24 | 0.089 | 1.50 | 1.64 | 11.5× | 1.5× | 2.0× | 64×32 (4/3) |
| 64 | 8192 | 3.74 | 8.25 | 12.02 | 0.221 | 5.92 | 6.14 | 16.9× | 1.4× | 2.0× | 64×32 (4/3) |
| 64 | 16384 | 14.84 | 32.66 | 47.91 | 0.718 | 23.22 | 26.17 | 20.7× | 1.4× | 1.8× | 64×32 (4/3) |
| 64 | 32768 | 59.28 | 130.10 | 191.12 | 2.65 | 98.84 | 98.70 | 22.3× | 1.3× | 1.9× | 64×32 (4/3) |
| 64 | 65536 | OOM | OOM | OOM | 10.75 | OOM | OOM | – | – | – | 128×32 (8/2) |
| 128 | 128 | 0.038 | 0.068 | 0.344 | 0.009 | 0.050 | 0.078 | 4.1× | 1.3× | 4.4× | 32×64 (4/3) |
| 128 | 256 | 0.040 | 0.073 | 0.401 | 0.012 | 0.068 | 0.096 | 3.2× | 1.1× | 4.2× | 32×64 (4/3) |
| 128 | 512 | 0.046 | 0.088 | 0.395 | 0.018 | 0.092 | 0.108 | 2.5× | 1.0× | 3.7× | 32×64 (4/3) |
| 128 | 1024 | 0.078 | 0.170 | 0.432 | 0.032 | 0.199 | 0.226 | 2.5× | 0.9× | 1.9× | 32×64 (4/3) |
| 128 | 2048 | 0.269 | 0.608 | 0.882 | 0.059 | 0.659 | 0.724 | 4.5× | 0.9× | 1.2× | 32×64 (4/3) |
| 128 | 4096 | 1.03 | 2.22 | 3.27 | 0.160 | 2.22 | 2.41 | 6.4× | 1.0× | 1.4× | 32×64 (4/3) |
| 128 | 8192 | 3.85 | 8.29 | 12.12 | 0.522 | 8.46 | 9.02 | 7.4× | 1.0× | 1.3× | 128×64 (8/3) |
| 128 | 16384 | 14.99 | 32.81 | 48.79 | 1.63 | 33.72 | 36.04 | 9.2× | 1.0× | 1.4× | 128×64 (8/2) |
| 128 | 32768 | 60.17 | 131.79 | 193.15 | 5.81 | 138.48 | 140.10 | 10.4× | 1.0× | 1.4× | 128×64 (8/2) |
| 128 | 65536 | OOM | OOM | OOM | 21.61 | OOM | OOM | – | – | – | 128×64 (8/2) |

### float32

| d | seq | pytorch fwd | pytorch bwd | pytorch e2e | flash fwd | flash bwd | flash e2e | fwd speedup | bwd speedup | e2e speedup | tiles Bq×Bk (warps/stages) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 16 | 128 | 0.038 | 0.072 | 0.389 | 0.006 | 0.038 | 0.041 | 6.2× | 1.9× | 9.5× | 32×64 (4/3) |
| 16 | 256 | 0.042 | 0.084 | 0.349 | 0.008 | 0.045 | 0.050 | 5.1× | 1.9× | 7.0× | 32×64 (4/3) |
| 16 | 512 | 0.053 | 0.104 | 0.368 | 0.011 | 0.056 | 0.065 | 4.7× | 1.9× | 5.7× | 32×64 (4/3) |
| 16 | 1024 | 0.116 | 0.270 | 0.382 | 0.018 | 0.111 | 0.126 | 6.3× | 2.4× | 3.0× | 32×64 (4/3) |
| 16 | 2048 | 0.459 | 1.06 | 1.51 | 0.032 | 0.339 | 0.370 | 14.3× | 3.1× | 4.1× | 32×64 (4/3) |
| 16 | 4096 | 1.74 | 3.93 | 5.67 | 0.062 | 1.19 | 1.26 | 27.8× | 3.3× | 4.5× | 64×64 (4/3) |
| 16 | 8192 | 6.66 | 15.31 | 21.96 | 0.158 | 4.48 | 4.68 | 42.2× | 3.4× | 4.7× | 64×64 (4/3) |
| 16 | 16384 | 26.42 | 60.71 | 87.06 | 0.502 | 17.70 | 18.18 | 52.6× | 3.4× | 4.8× | 64×64 (4/3) |
| 16 | 32768 | 105.60 | OOM | OOM | 1.89 | 70.99 | 72.72 | 55.9× | – | – | 64×64 (4/3) |
| 16 | 65536 | OOM | OOM | OOM | 6.51 | OOM | OOM | – | – | – | 128×128 (4/3) |
| 32 | 128 | 0.039 | 0.073 | 0.335 | 0.007 | 0.039 | 0.045 | 5.4× | 1.9× | 7.5× | 32×64 (4/3) |
| 32 | 256 | 0.041 | 0.079 | 0.382 | 0.010 | 0.044 | 0.051 | 4.0× | 1.8× | 7.5× | 32×64 (4/3) |
| 32 | 512 | 0.053 | 0.104 | 0.377 | 0.014 | 0.055 | 0.068 | 3.7× | 1.9× | 5.6× | 32×64 (4/3) |
| 32 | 1024 | 0.118 | 0.280 | 0.391 | 0.025 | 0.116 | 0.136 | 4.8× | 2.4× | 2.9× | 32×64 (4/3) |
| 32 | 2048 | 0.465 | 1.07 | 1.54 | 0.045 | 0.357 | 0.404 | 10.3× | 3.0× | 3.8× | 32×64 (4/3) |
| 32 | 4096 | 1.77 | 3.99 | 5.75 | 0.095 | 1.25 | 1.38 | 18.5× | 3.2× | 4.2× | 64×64 (4/3) |
| 32 | 8192 | 6.80 | 15.46 | 22.24 | 0.248 | 4.99 | 5.25 | 27.5× | 3.1× | 4.2× | 64×64 (4/3) |
| 32 | 16384 | 27.01 | 61.31 | 88.22 | 0.842 | 19.55 | 20.10 | 32.1× | 3.1× | 4.4× | 64×32 (4/3) |
| 32 | 32768 | 108.00 | OOM | OOM | 3.07 | 74.47 | 78.49 | 35.1× | – | – | 64×64 (4/3) |
| 32 | 65536 | OOM | OOM | OOM | 12.32 | OOM | OOM | – | – | – | 64×32 (4/2) |
| 64 | 128 | 0.040 | 0.073 | 0.388 | 0.009 | 0.040 | 0.046 | 4.3× | 1.8× | 8.4× | 32×64 (4/3) |
| 64 | 256 | 0.042 | 0.083 | 0.397 | 0.012 | 0.048 | 0.058 | 3.4× | 1.7× | 6.8× | 32×64 (4/3) |
| 64 | 512 | 0.055 | 0.182 | 0.567 | 0.018 | 0.063 | 0.078 | 3.1× | 2.9× | 7.3× | 32×64 (4/3) |
| 64 | 1024 | 0.129 | 0.301 | 0.433 | 0.031 | 0.141 | 0.168 | 4.2× | 2.1× | 2.6× | 32×64 (4/3) |
| 64 | 2048 | 0.495 | 1.12 | 1.63 | 0.057 | 0.438 | 0.504 | 8.6× | 2.6× | 3.2× | 32×64 (4/3) |
| 64 | 4096 | 1.87 | 4.15 | 6.01 | 0.138 | 1.55 | 1.71 | 13.5× | 2.7× | 3.5× | 64×64 (4/3) |
| 64 | 8192 | 7.21 | 16.15 | 23.36 | 0.413 | 6.12 | 6.48 | 17.5× | 2.6× | 3.6× | 64×64 (4/3) |
| 64 | 16384 | 28.39 | 63.45 | 91.95 | 1.44 | 24.38 | 27.24 | 19.8× | 2.6× | 3.4× | 64×64 (4/3) |
| 64 | 32768 | 115.40 | OOM | OOM | 6.17 | 124.82 | 104.57 | 18.7× | – | – | 64×32 (4/2) |
| 64 | 65536 | OOM | OOM | OOM | 23.42 | OOM | OOM | – | – | – | 128×32 (4/2) |
| 128 | 128 | 0.041 | 0.074 | 0.321 | 0.012 | 0.044 | 0.054 | 3.3× | 1.7× | 5.9× | 32×32 (4/3) |
| 128 | 256 | 0.046 | 0.087 | 0.380 | 0.018 | 0.058 | 0.073 | 2.5× | 1.5× | 5.2× | 32×32 (4/3) |
| 128 | 512 | 0.065 | 0.126 | 0.398 | 0.031 | 0.083 | 0.113 | 2.1× | 1.5× | 3.5× | 32×32 (4/3) |
| 128 | 1024 | 0.146 | 0.339 | 0.488 | 0.056 | 0.190 | 0.243 | 2.6× | 1.8× | 2.0× | 32×32 (4/3) |
| 128 | 2048 | 0.578 | 1.28 | 1.88 | 0.108 | 0.660 | 0.791 | 5.4× | 1.9× | 2.4× | 32×32 (4/3) |
| 128 | 4096 | 2.16 | 4.66 | 6.82 | 0.305 | 2.26 | 2.59 | 7.1× | 2.1× | 2.6× | 32×32 (4/3) |
| 128 | 8192 | 8.16 | 17.94 | 26.09 | 1.04 | 8.66 | 9.79 | 7.8× | 2.1× | 2.7× | 32×32 (4/3) |
| 128 | 16384 | 32.33 | 70.38 | 103.04 | 3.26 | 34.87 | 38.71 | 9.9× | 2.0× | 2.7× | 128×32 (8/2) |
| 128 | 32768 | 132.84 | OOM | OOM | 12.45 | 140.82 | 157.90 | 10.7× | – | – | 128×32 (8/2) |
| 128 | 65536 | OOM | OOM | OOM | 47.21 | OOM | OOM | – | – | – | 128×32 (8/2) |

The flash fp32 d=64 seq=32768 row has bwd > e2e. It sits next to the 24 GB limit, where the allocator is noisy. An
earlier run of the same config gave bwd 101.6 / e2e 103.7.

**Takeaways.**
- **The Triton forward wins everywhere.** It is 2–6× faster at N ≤ 1024, where PyTorch is bound by launches and
  allocations, and 9–56× faster at N ≥ 16k. The gap is largest at small d, where eager attention is pure N²
  memory traffic (scores, mask, softmax) and the fused kernel never writes S or P to HBM. At d = 128 the kernel
  becomes compute-bound: bf16 at N = 65536 takes 21.6 ms for 2·N²·d ≈ 1.1 TFLOP (causal half), which is ~51 TFLOP/s,
  ~40% of the A10's 125 TFLOP/s bf16 peak. fp32 (TF32 `tl.dot`) reaches ~23 TFLOP/s and is about 2× slower than
  bf16 at the same shape.
- **The backward dominates, and it is barely faster than eager.** The compiled backward recomputes P with full
  N × N fp32 matrices, so it is still O(N²) in memory traffic. Flash bwd is ≥ 90% of flash e2e at N ≥ 4096. Against
  eager it is 0.9–1.8× in bf16 and 1.5–3.4× in fp32. In bf16 at d = 128 it is no faster (0.9–1.0×), because it
  upcasts everything to fp32 (bf16 and fp32 flash bwd times are almost the same) while eager runs bf16 matmuls. So
  e2e speedups at N = 16384 are only 1.4–2.6× (bf16) and 2.7–4.8× (fp32). A tiled Triton backward (Algorithm 2) is
  the obvious next step.
- **Memory.** Eager PyTorch OOMs in the forward at N = 65536 (the N² bool mask plus N² scores and probabilities) and
  in fp32 backward from N = 32768. The flash forward runs at every size, because it only stores O(N·d) (Q, K, V, O, L).
  Only its compiled backward OOMs at N = 65536 (one fp32 N × N matrix is 16 GiB).
- **Small sequences are overhead.** Below ~1024 tokens all flash forward times are 6–60 µs. Eager e2e is ~0.35 ms
  regardless of size, which is CPU launch/autograd overhead: do_bench sees it because the GPU waits for the CPU.
- **Causal skipping.** Stopping the key loop at the diagonal makes the causal forward 1.4–1.5× faster than the full
  non-causal loop (N = 16384, d ∈ {16, 128}, both dtypes). This is less than 2× because query tiles near the start
  finish early, which leaves SMs idle at the end of the launch.
- **Tiles.** Autotuning picks 32×64 for short sequences (more programs to fill the 72 SMs). It picks 128×128 at
  d = 16 and long N (bf16), and 128×32–64 with 8 warps at d = 128. fp32 gets smaller tiles than bf16 at the same
  shape, because twice the bytes per element means shared memory limits the tile and stage count.

Raw data: `results/data/4_2_flash_benchmarking.jsonl` (one row per impl × dtype × d × N, with a `tile_config` field
for flash). Log: `results/data/4_2_flash_benchmarking.log`. Commands: `results/notes/4_2_flash_benchmarking.md`.
