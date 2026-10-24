# 4.1.1 pytorch_attention

Naive `cs336_basics.model.scaled_dot_product_attention`, fp32, batch 8, single head, no mask, on an A10 (22.5 GiB
usable). Mean of 100 timed passes after 10 warm-up passes; "mem" is `torch.cuda.memory_allocated()` right after the
forward pass, before backward.

| d_model | seq_len | forward (ms) | backward (ms) | mem before backward (MiB) |
|--------:|--------:|-------------:|--------------:|--------------------------:|
|  16 |   256 |  0.22 |   0.46 |   20.9 |
|  16 |  1024 |  0.99 |   2.47 |   82.8 |
|  16 |  4096 | 14.63 |  37.12 | 1050.6 |
|  16 |  8192 | 57.89 | 147.49 | 4133.0 |
|  16 | 16384 | OOM (forward) | – | – |
|  32 |   256 |  0.22 |   0.46 |   21.5 |
|  32 |  1024 |  1.08 |   2.55 |   85.3 |
|  32 |  4096 | 15.05 |  37.75 | 1060.6 |
|  32 |  8192 | 59.62 | 149.98 | 4153.0 |
|  32 | 16384 | OOM (forward) | – | – |
|  64 |   256 |  0.22 |   0.46 |   22.8 |
|  64 |  1024 |  1.12 |   2.62 |   90.3 |
|  64 |  4096 | 15.42 |  38.43 | 1080.6 |
|  64 |  8192 | 61.06 | 152.03 | 4193.0 |
|  64 | 16384 | OOM (forward) | – | – |
| 128 |   256 |  0.34 |   0.84 |   25.3 |
| 128 |  1024 |  1.24 |   2.86 |  100.3 |
| 128 |  4096 | 16.95 |  40.16 | 1120.6 |
| 128 |  8192 | 67.17 | 160.42 | 4273.0 |
| 128 | 16384 | OOM (forward) | – | – |

**Memory accounting (smallest OOM config: d_model = 16, seq_len = 16384).** One fp32 score-sized tensor is
B · N² · 4 bytes = 8 · 16384² · 4 = 8 GiB, while Q, K, V and O are only 8 · 16384 · 16 · 4 = 8 MiB each. Autograd
saves Q, K, V, the softmax's `exp(x − max)` and the attention weights P. That is 2 score-sized tensors (16 GiB)
plus 32 MiB of inputs and a few B·N-sized row statistics. The forward pass needs even more: the Python locals in
`scaled_dot_product_attention` and `softmax` keep the scaled scores and `x − max` alive until the softmax returns, so
4 score-sized tensors (32 GiB) are live at once. That is far more than the 22.5 GiB available, so it fails already
in the forward pass. The measurements at N = 8192 match this: the forward peak is 8213 MiB ≈ 4 × 2 GiB, 4133 MiB
(≈ 2 × 2 GiB + inputs) stays allocated for backward, and the backward peak is ≈ 6 × 2 GiB, because dP and dS are
also N² in size.

**Answer.** Every configuration up to seq_len 8192 runs, and every configuration at 16384 runs out of memory,
whatever d_model is. Time and memory are both dominated by the N × N score matrix: going from 4096 to 8192 multiplies
forward time, backward time and saved memory by about 4×. Going from d_model 16 to 128 adds only about 15%, because
the matmuls with V and Kᵀ are cheap next to the elementwise softmax passes over B·N² elements, which are
memory-bandwidth-bound. Backward takes about 2.5× as long as forward. The memory saved for backward grows
quadratically in the sequence length (2 · B · N² · 4 bytes plus O(B·N·d)), so doubling N quadruples it. To remove
this cost, never materialize the N × N matrix. Compute attention tile by tile with an online softmax (FlashAttention),
save only Q, K, V, O and the per-row logsumexp L (O(B·N)), and recompute the score tiles from Q, K and L in the
backward pass. A fused kernel also avoids reading and writing N² intermediates to HBM, so it is faster as well.

Raw data: `results/data/4_1_pytorch_attention.jsonl` (log: `results/data/4_1_pytorch_attention.log`).
