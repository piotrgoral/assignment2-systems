# 3.2 gradient_checkpointing (b)

**Answer.** The best single-level strategy is to **checkpoint every Transformer block on its own** (segments of 1 block). Its measured peak is **9085 MiB** (8263 MiB of activations), versus 11455 MiB with 2 blocks per checkpoint and 10765 MiB with sublayer checkpointing. The reasoning: with segments of k blocks, the peak is about (N/k)·C + k·R. Here C = 80 MiB is one checkpoint (the 4×2048×2560 fp32 input of a segment), and R ≈ 3.6 GiB is the residual memory of one block (handout: 3651 MiB). This is minimized at k* = √(N·C/R) = √(32·80/3651) ≈ 0.84, so among whole-block sizes, k = 1 wins. A block's residuals are 45× bigger than a checkpoint, so it pays to store many cheap checkpoints and recompute as little as possible at a time. The next smaller granularity checkpoints the attention and the feed-forward sublayers separately. It doubles the checkpoints to 5 GiB but saves less than that on the transient memory, because the attention sublayer (its 4×16×2048×2048 score/probability tensors) dominates one block's residuals. The next larger size, k = 2, halves the checkpoints but recomputes two blocks' worth of residuals at once. Every option costs the same one extra forward per block, and step times are within ~5%.

xl block (d_model 2560, d_ff 10240, 16 heads), batch 4, ctx 2048, N = 32 blocks, fp32, compiled; one step = forward + backward; A10 (24 GiB):

| strategy | checkpoints | saved outside checkpoints (MiB) | peak (MiB) | activation peak (MiB) | step (s) |
|---|--:|--:|--:|--:|--:|
| none | 0 | — | OOM (>22.9 GiB) | — | — |
| sublayer (attn / FFN separately) | 64 | 5120 | 10765 | 9943 | 20.9 ± 0.1 |
| **1 block / checkpoint** | 32 | 2560 | **9085** | **8263** | 21.4 ± 0.2 |
| 2 blocks / checkpoint | 16 | 1280 | 11455 | 10633 | 21.8 ± 0.1 |
| 4 blocks / checkpoint | 8 | 640 | 18115 | 17293 | 21.9 ± 0.1 |
| 8 blocks / checkpoint | 4 | 320 | OOM (>23.2 GiB) | — | — |

The winner's peak = 822 MiB params + grads + 2560 MiB of checkpoints + ~5.7 GiB transient while one block is recomputed and back-propagated (3.6 GiB of residuals plus backward temporaries such as the gradient of the attention probabilities). Going from k = 1 → 2 → 4, the peak rises by 2370 and 6660 MiB. Adding back the checkpoint memory saved (1280 and 640 MiB) gives exactly 3650 MiB per extra block in a segment, which is R, the k·R term.

Setup note: the full xl LM (~3.4B params, ~27 GB of fp32 params + grads) cannot run on a 24 GiB A10. As in the handout's `four_blocks` example, one block is applied 32 times, so the peak is almost all activations. Peak = `torch.cuda.max_memory_allocated()` over 3 measured steps after 1 warm-up step. A memory snapshot of the winner is in `results/data/3_2_gradient_checkpointing_b_snapshot.pickle` (open with pytorch.org/memory_viz).

Raw data: `results/data/3_2_gradient_checkpointing_b.jsonl` (log: `results/data/3_2_gradient_checkpointing_b.log`).
