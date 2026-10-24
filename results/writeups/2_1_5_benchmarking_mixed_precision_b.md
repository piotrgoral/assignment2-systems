# 2.1.5 benchmarking_mixed_precision (b)

**Answer.** The sensitive parts of layer normalization are its reductions: the mean and especially the variance (a sum
of squares), followed by `1/sqrt(var + eps)`. In FP16 the sum of squares can overflow past 65504, small variances and
`eps` (1e-5) can underflow or lose precision, and accumulating many terms with 10 mantissa bits causes rounding error
(as in the accumulation problem), so autocast runs LayerNorm in FP32. BF16 has FP32's 8-bit exponent, so overflow and
underflow mostly go away, but with only 7 mantissa bits the rounding error in the mean/variance is even worse than
FP16's. So we still want LayerNorm's statistics computed in FP32, and PyTorch's autocast does that for BF16 too (our
BF16 probe shows `ln` output `float32`; the cs336_basics `RMSNorm` likewise upcasts to fp32 internally).

Raw data: `results/data/2_1_5_benchmarking_mixed_precision_a.jsonl`.
