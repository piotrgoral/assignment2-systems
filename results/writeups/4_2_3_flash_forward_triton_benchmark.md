# 4.2.2 flash_forward (b), extra: benchmarking the Triton FlashAttention-2 forward

Not a deliverable. This runs the same grid as 4.1.1 and the pure-PyTorch flash benchmark
(`4_2_2_flash_forward_pytorch_benchmark.md`) for the fused Triton kernel (`FlashAttentionTriton`). Setup: fp32,
batch 8, single head, no mask, on an A10, with 10 warm-up passes and the mean of 100 timed forward passes. Only the
forward pass is timed (the benchmark's `flash_triton` impl is forward-only). fp32 `tl.dot` runs in true IEEE fp32,
not TF32. Square tiles Bq = Bk ∈ {16, 32, 64, 128}, with the kernel's default `num_warps` / `num_stages`; nothing else is tuned. The eager, compiled and flash-PyTorch columns (tile 128) are
copied from `4_1_pytorch_attention.md`, `4_2_torch_compile_a.md` and the flash-PyTorch writeup. Times are in ms. In
each row, the fastest Triton tile is in bold. "smem" means the kernel needs more shared memory than the A10's 99 KiB
per block, so it does not launch. "triton mem" is `torch.cuda.memory_allocated()` after a forward pass with its graph
alive. It is the same for every tile size.

| d_model | seq_len | eager | compiled | flash pytorch | triton 16 | triton 32 | triton 64 | triton 128 | best triton vs eager | vs compiled | eager mem (MiB) | triton mem (MiB) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 256 | 0.22 | 0.19 | 1.06 | 0.10 | **0.09** | 0.09 | smem | 2× | 2.0× | 20.9 | 0.6 |
| 16 | 1024 | 0.99 | 0.45 | 12.38 | 0.28 | 0.23 | **0.16** | smem | 6× | 2.8× | 82.8 | 2.5 |
| 16 | 4096 | 14.63 | 6.41 | 184.18 | 3.41 | 2.61 | **1.71** | smem | 9× | 3.7× | 1050.6 | 10.1 |
| 16 | 8192 | 57.89 | 25.50 | 666.10 | 13.30 | 10.11 | **6.44** | smem | 9× | 4.0× | 4133.0 | 20.2 |
| 16 | 16384 | OOM | OOM | 2639.24 | 53.19 | 39.75 | **24.94** | smem | – | – | OOM | 40.5 |
| 32 | 256 | 0.22 | 0.19 | 1.06 | 0.10 | 0.09 | **0.09** | smem | 2× | 2.0× | 21.5 | 1.3 |
| 32 | 1024 | 1.08 | 0.47 | 12.32 | 0.55 | 0.32 | **0.24** | smem | 5× | 2.0× | 85.3 | 5.0 |
| 32 | 4096 | 15.05 | 6.90 | 174.05 | 7.41 | 4.17 | **3.06** | smem | 5× | 2.3× | 1060.6 | 20.1 |
| 32 | 8192 | 59.62 | 27.27 | 681.02 | 29.28 | 16.11 | **11.56** | smem | 5× | 2.4× | 4153.0 | 40.2 |
| 32 | 16384 | OOM | OOM | 2666.25 | 117.15 | 63.63 | **44.95** | smem | – | – | OOM | 80.5 |
| 64 | 256 | 0.22 | 0.19 | 0.99 | 0.12 | **0.10** | 0.13 | smem | 2× | 1.9× | 22.8 | 2.5 |
| 64 | 1024 | 1.12 | 0.53 | 11.37 | 0.95 | **0.53** | 0.78 | smem | 2× | 1.0× | 90.3 | 10.0 |
| 64 | 4096 | 15.42 | 7.55 | 166.78 | 13.21 | **7.73** | 12.32 | smem | 2× | 1.0× | 1080.6 | 40.1 |
| 64 | 8192 | 61.06 | 30.05 | 652.66 | 52.02 | **30.08** | 48.29 | smem | 2× | 1.0× | 4193.0 | 80.2 |
| 64 | 16384 | OOM | OOM | 2735.52 | 208.28 | **117.73** | 193.06 | smem | – | – | OOM | 160.5 |
| 128 | 256 | 0.34 | 0.19 | 1.19 | 0.17 | **0.15** | smem | smem | 2× | 1.2× | 25.3 | 5.0 |
| 128 | 1024 | 1.24 | 0.61 | 11.49 | 1.74 | **1.51** | smem | smem | 1× | 0.4× | 100.3 | 20.0 |
| 128 | 4096 | 16.95 | 9.11 | 173.01 | 25.15 | **22.95** | smem | smem | 1× | 0.4× | 1120.6 | 80.1 |
| 128 | 8192 | 67.17 | 38.43 | 671.05 | 98.87 | **94.91** | smem | smem | 1× | 0.4× | 4273.0 | 160.2 |
| 128 | 16384 | OOM | OOM | 2659.70 | 395.77 | **391.44** | smem | smem | – | – | OOM | 320.5 |

The compiled run at seq_len 16384 got through the forward pass but ran out of memory in backward, so 4.2 recorded no
forward time for it.

**Takeaways.**
- **Fusing fixes memory, and speed only when the matmuls are cheap.** The kernel runs true fp32 `tl.dot`
  (`input_precision="ieee"`), with no tensor cores, just like the eager and compiled baselines. From seq_len 1024 up,
  the best tile is 5–9× faster than eager and 2–4× faster than `torch.compile` at d = 16–32, where attention is
  memory-bound. At d = 64 it is ~2× faster than eager but only level with compiled, and at d = 128 it is level with
  eager and ~2.5× *slower* than compiled (0.4×). It is ~7–110× faster than the Python-loop flash version (the gap is
  largest at small d, where the PyTorch version's time is all launch overhead). The whole inner loop (both matmuls,
  the online-softmax rescaling) runs in one kernel per query tile. S and P̃ live only in registers and shared memory,
  so nothing of size N × N goes through HBM. Eager and compiled both write the B·N² score and probability tensors to
  HBM. That traffic is what limits them at small d, but at large d the FLOPs dominate, and cuBLAS's fp32 GEMMs do
  them faster than Triton's IEEE dot. With Triton's default TF32 dot (an earlier run) the same kernel was 1.7–10×
  faster (4–7× for the best tile at seq_len 16384), but that isn't a like-for-like comparison with fp32 baselines.
- **Memory stays O(B·N·d).** The memory numbers are the same as for the PyTorch flash version: Q, K, V, O, the
  benchmark's `grad_out` and L. The forward peak equals that, because no intermediate is allocated at all. The
  PyTorch flash numbers are ~8 MiB higher only because its matmuls allocate the cuBLAS workspace, which the Triton
  kernel never touches. seq_len 16384 runs for every d_model, and eager / compiled cannot do that.
- **Time grows as N².** From seq_len 4096 up, each doubling of N costs ~3.8–4.1× with the best tile, and time grows
  with d_model. Throughput at the largest size: 4·B·N²·d FLOPs gives ~5.5 TFLOP/s at d = 16 (tile 64), ~6.1 at
  d = 32 (tile 64), ~4.7 at d = 64 (tile 32) and ~2.8 at d = 128 (tile 32). That is 9–20% of the A10's ~31 TFLOP/s
  fp32 (non-tensor-core) peak. Below ~1024 tokens times are flat at ~0.09–0.15 ms, which is the fixed cost of
  `autograd.Function.apply` plus a kernel launch: at seq_len 256 there are only 8·256/Bq programs.
- **Tile size matters, and shared memory caps it.** The test's 16 × 16 tiles are up to 2.6× slower than the best
  tile at long sequences (d = 16–64), and about the same at d = 128. Bigger tiles mean fewer K/V reloads per query
  row and larger `tl.dot`s. The limit is shared memory. In fp32 the K and V tiles are 2·Bk·d·4 bytes, and Triton
  keeps several of them in flight to pipeline the loads (the default `num_stages`). Tile 128 needs more than the
  99 KiB limit at every d, and tile 64 does too at d = 128. The best tile is 64 at d = 16–32 and 32 at d = 64–128.
  At d = 64, tile 64 still launches but is ~1.6× slower than tile 32. Choosing tiles per d, using non-square
  Bq ≠ Bk and tuning `num_warps` / `num_stages` (e.g. with `triton.autotune`) would help, and bf16 inputs would
  halve the tile footprint and enable tensor cores. That belongs to the later flash_benchmarking problem.

Raw data: `results/data/4_2_3_flash_forward_triton.jsonl` (one row per config and tile size, `"tile_size"` field;
`"status": "out_of_shared_memory"` for the smem rows). Log: `results/data/4_2_3_flash_forward_triton.log`.
