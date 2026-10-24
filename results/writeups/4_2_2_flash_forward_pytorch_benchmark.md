# 4.2.2 flash_forward (a), extra: benchmarking the pure-PyTorch FlashAttention-2 forward

Not a deliverable. This is a sanity check of the tiled forward pass (`FlashAttentionPytorch`) on the same grid as
4.1.1. Setup: fp32, batch 8, single head, no mask, on an A10. Tiles are Bq = Bk = 128, with 2 warm-up passes and the
mean of 10 timed forward passes (fewer than 4.1.1's 10 + 100, because a pass at seq_len 16384 takes ~2.7 s). There is
no backward yet, so only the forward pass is timed. "mem" is `torch.cuda.memory_allocated()` after a forward pass with
its graph alive. For flash that is Q, K, V, O, L, plus the benchmark's unused `grad_out`, which is also counted in 4.1.1.
The eager columns are copied from `4_1_pytorch_attention.md`.

| d_model | seq_len | eager fwd (ms) | flash fwd (ms) | flash / eager | eager mem (MiB) | flash mem (MiB) |
|--------:|--------:|---------------:|---------------:|--------------:|----------------:|----------------:|
|  16 |   256 | 0.22 | 1.06 | 5× | 20.9 | 8.8 |
|  16 |  1024 | 0.99 | 12.38 | 12× | 82.8 | 10.7 |
|  16 |  4096 | 14.63 | 184.18 | 13× | 1050.6 | 18.2 |
|  16 |  8192 | 57.89 | 666.10 | 12× | 4133.0 | 28.4 |
|  16 | 16384 | OOM | 2639.24 | – | OOM | 48.6 |
|  32 |   256 | 0.22 | 1.06 | 5× | 21.5 | 9.4 |
|  32 |  1024 | 1.08 | 12.32 | 11× | 85.3 | 13.2 |
|  32 |  4096 | 15.05 | 174.05 | 12× | 1060.6 | 28.2 |
|  32 |  8192 | 59.62 | 681.02 | 11× | 4153.0 | 48.4 |
|  32 | 16384 | OOM | 2666.25 | – | OOM | 88.6 |
|  64 |   256 | 0.22 | 0.99 | 5× | 22.8 | 10.6 |
|  64 |  1024 | 1.12 | 11.37 | 10× | 90.3 | 18.2 |
|  64 |  4096 | 15.42 | 166.78 | 11× | 1080.6 | 48.2 |
|  64 |  8192 | 61.06 | 652.66 | 11× | 4193.0 | 88.4 |
|  64 | 16384 | OOM | 2735.52 | – | OOM | 168.6 |
| 128 |   256 | 0.34 | 1.19 | 3× | 25.3 | 13.1 |
| 128 |  1024 | 1.24 | 11.49 | 9× | 100.3 | 28.2 |
| 128 |  4096 | 16.95 | 173.01 | 10× | 1120.6 | 88.2 |
| 128 |  8192 | 67.17 | 671.05 | 10× | 4273.0 | 168.4 |
| 128 | 16384 | OOM | 2659.70 | – | OOM | 328.6 |

Tile size matters a lot (d_model 64): with the minimum tiles of 16 × 16 (the ones the test uses), a forward pass takes
42 ms at seq_len 256 and 679 ms at 1024, which is ~40–60× slower than with 128 × 128 tiles.

**Takeaways.**
- **Memory is fixed.** No N × N tensor is ever allocated, so every config at seq_len 16384 now runs. Saved memory is
  O(B·N·d): it doubles with N and with d_model (4 + 1 tensors of B·N·d, plus an L of B·N), instead of quadrupling with N.
  At seq_len 8192 and d_model 16 it is 28 MiB, against 4.1 GiB for eager. The forward peak is only ~2–3 MiB above that,
  i.e. one tile of scores.
- **Speed is not.** The Python version is ~10× slower than eager at long sequences. Every one of the
  (N/Bq)·(N/Bk) tile pairs launches about 8 small kernels (two matmuls, max, exp, sum, rescales). At seq_len 16384
  that is 16384 pairs, ~0.16 ms each. Each kernel is too small to fill the GPU, so time depends only on the number of
  tiles: it is flat in d_model and grows 4× per doubling of N. Bigger tiles mean fewer launches, hence the tile-size
  effect above. Fusing the whole inner loop into one kernel per query tile is what the Triton version in (b) is for.
- **Why not torch.compile.** Dynamo would trace the Python tile loops and unroll them into one graph with
  (N/Bq)·(N/Bk) iterations. That is ~16k matmul pairs at seq_len 16384 with 128 tiles, and ~1M with 16 tiles, so
  compiling would take far longer than running, and it would be redone for every shape. Compiling the inner tile step
  on its own would still launch it from Python for every tile pair. So the compile comparison is left for the Triton
  kernel.

Raw data: `results/data/4_2_2_flash_forward_pytorch.jsonl` (log: `results/data/4_2_2_flash_forward_pytorch.log`). The
tile-16 rows have `"tile_size": 16`.
