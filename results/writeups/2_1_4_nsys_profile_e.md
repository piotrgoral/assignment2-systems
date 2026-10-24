# 2.1.4 nsys_profile (e)

**Answer.** Inside self-attention, the softmax takes **1.4–1.9× longer** on the GPU than the two attention matmuls (QKᵀ and PV) put together (small ctx 512: 846 µs vs 491 µs per layer), even though it has **~51× fewer FLOPs** (≈5·T² vs 4·T²·d_h per head, d_h = 64). Per FLOP, it is therefore ~70–100× slower. The matmuls are compute-bound and reach 5–7 TFLOP/s; the softmax is memory-bound, running at ~0.07 TFLOP/s but moving ~450–490 GB/s, close to the A10's 600 GB/s. Its five kernels (max, subtract, exp, sum, divide) each make a full pass over the fp32 B×H×T×T score tensor.

Per-layer GPU kernel time of one attention call (`nvtx_kern_sum` of the `forward`-mode profiles, summed over the 3 profiled steps and ÷ (3 × layers)). "QKᵀ GEMM" and "scale + mask" are the GEMM and the non-GEMM kernels inside `computing attention scores`; "softmax" = `computing softmax`; "PV GEMM" = `final matmul`.

| size | ctx | bs | QKᵀ GEMM | scale + mask | softmax | PV GEMM | softmax ÷ (QKᵀ+PV) time | (QKᵀ+PV) ÷ softmax FLOPs | matmul TFLOP/s | softmax TFLOP/s | softmax GB/s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| small | 256 | 4 | 54 µs | 101 µs | 228 µs | 105 µs | 1.44× | 51× | 5.1 | 0.069 | 441 |
| small | 512 | 4 | 190 µs | 375 µs | 846 µs | 301 µs | 1.72× | 51× | 6.6 | 0.074 | 476 |
| small | 1024 | 4 | 637 µs | 1587 µs | 3301 µs | 1101 µs | 1.90× | 51× | 7.4 | 0.076 | 488 |
| medium | 256 | 4 | 78 µs | 131 µs | 303 µs | 123 µs | 1.50× | 51× | 5.3 | 0.069 | 443 |
| medium | 512 | 4 | 257 µs | 499 µs | 1122 µs | 421 µs | 1.65× | 51× | 6.3 | 0.075 | 479 |
| medium | 1024 | 2 | 449 µs | 1063 µs | 2213 µs | 767 µs | 1.82× | 51× | 7.1 | 0.076 | 485 |

FLOPs per attention call: QKᵀ and PV each 2·B·H·T²·d_h; softmax ≈ 5·B·H·T² (max, subtract, exp, sum, divide, one FLOP per element each). The ratio is 0.8·d_h ≈ 51 for both sizes (d_h = 768/12 = 1024/16 = 64), independent of T, because both scale as T². "softmax GB/s" assumes 8 fp32 passes over the T×T tensor: max and sum read it once, while subtract, exp and divide each read and write it. Peak A10 numbers for reference: 31.2 TFLOP/s fp32, 600 GB/s.

Counting the scale + mask kernels too, the attention's non-matmul work takes 2.1–2.8× the matmul time (small ctx 512: 375 + 846 µs vs 491 µs, 2.5×). This is the overhead that a fused (Flash) attention kernel removes. Note that the `nvtx_sum` ranges (e.g. `computing softmax` ≈ 109 µs avg in small ctx 512) measure only the CPU time spent launching the kernels, not GPU time, so they cannot be used for this comparison.

Raw data: `results/data/2_1_4_nsys_profile/*_forward_nvtx_kern_sum.csv`.
