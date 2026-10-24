# 2.1.5 benchmarking_mixed_precision (a)

**Answer.** With fp32 params and FP16 autocast:

| component | dtype |
|---|---|
| model parameters inside the autocast context | `float32` (autocast casts op inputs, not the stored weights) |
| output of `fc1` | `float16` |
| output of `ln` (LayerNorm) | `float32` |
| predicted logits (`fc2` output) | `float16` |
| loss (`cross_entropy`) | `float32` |
| model gradients | `float32` (same dtype as the params) |

The ReLU output is also `float16`, since ReLU just keeps its input's dtype. The BF16 run gives the same table with
`bfloat16` in place of `float16`.

Raw data: `results/data/2_1_5_benchmarking_mixed_precision_a.jsonl` (log: `results/data/2_1_5_benchmarking_mixed_precision_a.log`).
