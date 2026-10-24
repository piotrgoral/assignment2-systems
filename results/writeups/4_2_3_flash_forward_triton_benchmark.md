# 4.2.2 flash_forward (b), extra: benchmarking the Triton FlashAttention-2 forward

Not a deliverable. This runs the same grid as 4.1.1 and the pure-PyTorch flash benchmark
(`4_2_2_flash_forward_pytorch_benchmark.md`) for the fused Triton kernel (`FlashAttentionTriton`). Setup: fp32,
batch 8, single head, no mask, on an A10, with 10 warm-up passes and the mean of 100 timed forward passes. Only the
forward pass is timed, because there is no backward yet. Square tiles Bq = Bk ∈ {16, 32, 64, 128}, with the kernel's
default `num_warps` / `num_stages`; nothing else is tuned. The eager, compiled and flash-PyTorch columns (tile 128) are
copied from `4_1_pytorch_attention.md`, `4_2_torch_compile_a.md` and the flash-PyTorch writeup. Times are in ms. In
each row, the fastest Triton tile is in bold. "smem" means the kernel needs more shared memory than the A10's 99 KiB
per block, so it does not launch. "triton mem" is `torch.cuda.memory_allocated()` after a forward pass with its graph
alive. It is the same for every tile size.

| d_model | seq_len | eager | compiled | flash pytorch | triton 16 | triton 32 | triton 64 | triton 128 | best triton vs eager | vs compiled | eager mem (MiB) | triton mem (MiB) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 256 | 0.22 | 0.19 | 1.06 | 0.09 | 0.09 | 0.09 | **0.09** | 2× | 2.1× | 20.9 | 0.6 |
| 16 | 1024 | 0.99 | 0.45 | 12.38 | 0.18 | 0.11 | 0.09 | **0.09** | 11× | 5.0× | 82.8 | 2.5 |
| 16 | 4096 | 14.63 | 6.41 | 184.18 | 2.03 | 0.83 | 0.44 | **0.38** | 38× | 16.8× | 1050.6 | 10.1 |
| 16 | 8192 | 57.89 | 25.50 | 666.10 | 7.97 | 3.27 | 1.61 | **1.47** | 39× | 17.4× | 4133.0 | 20.2 |
| 16 | 16384 | OOM | OOM | 2639.24 | 31.96 | 12.81 | 6.25 | **5.66** | – | – | OOM | 40.5 |
| 32 | 256 | 0.22 | 0.19 | 1.06 | **0.09** | 0.09 | 0.09 | 0.09 | 2× | 2.2× | 21.5 | 1.3 |
| 32 | 1024 | 1.08 | 0.47 | 12.32 | 0.21 | 0.14 | **0.10** | 0.12 | 11× | 4.6× | 85.3 | 5.0 |
| 32 | 4096 | 15.05 | 6.90 | 174.05 | 2.48 | 1.34 | **0.71** | 1.02 | 21× | 9.7× | 1060.6 | 20.1 |
| 32 | 8192 | 59.62 | 27.27 | 681.02 | 9.73 | 5.24 | **2.71** | 3.96 | 22× | 10.1× | 4153.0 | 40.2 |
| 32 | 16384 | OOM | OOM | 2666.25 | 38.67 | 20.57 | **10.65** | 14.88 | – | – | OOM | 80.5 |
| 64 | 256 | 0.22 | 0.19 | 0.99 | **0.09** | 0.09 | 0.09 | smem | 2× | 2.1× | 22.8 | 2.5 |
| 64 | 1024 | 1.12 | 0.53 | 11.37 | 0.27 | 0.18 | **0.14** | smem | 8× | 3.8× | 90.3 | 10.0 |
| 64 | 4096 | 15.42 | 7.55 | 166.78 | 3.64 | 2.00 | **1.32** | smem | 12× | 5.7× | 1080.6 | 40.1 |
| 64 | 8192 | 61.06 | 30.05 | 652.66 | 14.22 | 7.73 | **5.05** | smem | 12× | 5.9× | 4193.0 | 80.2 |
| 64 | 16384 | OOM | OOM | 2735.52 | 57.50 | 30.62 | **19.71** | smem | – | – | OOM | 160.5 |
| 128 | 256 | 0.34 | 0.19 | 1.19 | 0.09 | **0.09** | smem | smem | 4× | 2.2× | 25.3 | 5.0 |
| 128 | 1024 | 1.24 | 0.61 | 11.49 | 0.45 | **0.28** | smem | smem | 4× | 2.2× | 100.3 | 20.0 |
| 128 | 4096 | 16.95 | 9.11 | 173.01 | 6.65 | **3.83** | smem | smem | 4× | 2.4× | 1120.6 | 80.1 |
| 128 | 8192 | 67.17 | 38.43 | 671.05 | 26.20 | **14.77** | smem | smem | 5× | 2.6× | 4273.0 | 160.2 |
| 128 | 16384 | OOM | OOM | 2659.70 | 102.27 | **58.28** | smem | smem | – | – | OOM | 320.5 |

The compiled run at seq_len 16384 got through the forward pass but ran out of memory in backward, so 4.2 recorded no
forward time for it.

**Takeaways.**
- **Fusing fixes speed.** With the best tile, the Triton forward pass is 4–39× faster than eager and 2–17× faster
  than `torch.compile` from seq_len 1024 up. It is ~40–470× faster than the Python-loop flash version (the gap is largest at small d, where the PyTorch version's time is all launch overhead). The whole
  inner loop (both matmuls, the online-softmax rescaling) runs in one kernel per query tile. S and P̃ live only in
  registers and shared memory, so nothing of size N × N goes through HBM. Eager and compiled both write the B·N²
  score and probability tensors to HBM, and that traffic is what limits them.
- **Memory stays O(B·N·d).** The memory numbers are the same as for the PyTorch flash version: Q, K, V, O, the
  benchmark's `grad_out` and L. The forward peak equals that, because no intermediate is allocated at all. The
  PyTorch flash numbers are ~8 MiB higher only because its matmuls allocate the cuBLAS workspace, which the Triton
  kernel never touches. seq_len 16384 runs for every d_model, and eager / compiled cannot do that.
- **Time grows as N².** From seq_len 4096 up, each doubling of N costs ~3.9×, and at a fixed tile time grows with d_model,
  less than linearly at small d. Throughput at the largest sizes: 4·B·N²·d FLOPs gives ~24 TFLOP/s at d = 16 (tile 128), ~28 TFLOP/s
  at d = 64 (tile 64) and ~19 TFLOP/s at d = 128 (tile 32). The A10's TF32 tensor-core peak is ~62 TFLOP/s, so the
  best configs reach ~30–45% of it. Below ~1024 tokens everything is flat at ~0.09 ms, which is the fixed cost of
  `autograd.Function.apply` plus a kernel launch: at seq_len 256 there are only 8·256/Bq programs.
- **Tile size matters, and shared memory caps it.** The test's 16 × 16 tiles are 2–5× slower than the best tile at
  long sequences. Bigger tiles mean fewer K/V reloads per query row and larger, more efficient `tl.dot`s. The limit
  is shared memory. In fp32 the K and V tiles are 2·Bk·d·4 bytes, and Triton keeps several of them in flight to
  pipeline the loads (the default `num_stages`). Bq = Bk = 64 at d = 128 needs 177 KiB and 128 at d = 64 needs
  160 KiB, both over the 99 KiB limit. Smaller tiles help at large d (best: 128 at d = 16, 64 at d = 32–64, 32 at
  d = 128). Choosing tiles per d, using non-square Bq ≠ Bk and tuning `num_warps` / `num_stages` (e.g. with
  `triton.autotune`) would help, and bf16 inputs would halve the tile footprint. That belongs to the later
  flash_benchmarking problem.

Raw data: `results/data/4_2_3_flash_forward_triton.jsonl` (one row per config and tile size, `"tile_size"` field;
`"status": "out_of_shared_memory"` for the smem rows). Log: `results/data/4_2_3_flash_forward_triton.log`.
