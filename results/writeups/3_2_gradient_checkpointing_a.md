# 3.2 gradient_checkpointing (a)

**Answer.** Use recursive (binary) checkpointing. Split the N blocks into two halves and wrap each half in `checkpoint`. Inside each half, the function that `checkpoint` re-runs during backward does the same thing again: it splits its blocks in two and checkpoints each part, down to single blocks. At any moment during backward, only one checkpointed input per recursion level is alive, plus the residuals of the one block whose backward is running. That gives a peak activation memory of **O(log N)** checkpoints plus **O(1)** block residuals. When one block's residuals dominate the bookkeeping, this is effectively a constant (one block) plus log₂N tiny tensors, versus O(N) without checkpointing. The cost is compute: every block's forward is re-run once per recursion level, so a step costs **O(N log N)** block-forwards (about log₂N + 1 forwards per block, instead of 1 without checkpointing or 2 with one level).

```python
from torch.utils.checkpoint import checkpoint

def run(x, blocks):                     # blocks: a list of N modules
    if len(blocks) == 1:
        return blocks[0](x)
    mid = len(blocks) // 2
    x = checkpoint(run, x, blocks[:mid], use_reentrant=False)   # keeps only x; re-runs (recursively) in backward
    return checkpoint(run, x, blocks[mid:], use_reentrant=False)

y = run(x, model.layers)
```

Measured check (`--strategy recursive` vs. one checkpoint per block, xl block, batch 4, ctx 2048, A10). Each doubling of N adds exactly one 80 MiB checkpoint to the recursive peak, while the one-level peak grows by 80 MiB per block. The extra compute shows in the step time.

| N | recursive peak (MiB) | 1 block / checkpoint peak (MiB) | recursive step (s) | 1 block / checkpoint step (s) |
|--:|--:|--:|--:|--:|
| 8  | 6845 | 7165  | 6.9  | 5.4  |
| 16 | 6925 | 7805  | 15.4 | 10.9 |
| 32 | 7005 | 9085  | 34.0 | 21.8 |
| 64 | 7085 | 11645 | 74.6 | 44.0 |

Peaks include 822 MiB of params + grads of the one shared block. Raw data: `results/data/3_2_gradient_checkpointing_a.jsonl` (log: `results/data/3_2_gradient_checkpointing_a.log`).
