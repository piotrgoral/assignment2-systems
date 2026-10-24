# 2.1.4 nsys_profile (c)

**Answer.** Kernels that aren't matmuls take 28–48% of the forward pass's GPU time (small ctx 512: 41%), and nearly all of it is memory-bound elementwise kernels and reductions. The biggest is the attention softmax: its `max` and `sum` reductions plus its subtract, `exp` and divide kernels take 14% at ctx 512 and 22% at ctx 1024. Next come the score scaling (`Mul` by 1/√d_k) and the causal-mask `where` (3–10%), then the SwiGLU `sigmoid`/`Mul` (~7%), and the RoPE / RMSNorm `Mul`, `add`, `cat` and `pow`/`mean` kernels together with the residual adds (~10%). The attention parts operate on the (B, H, T, T) score tensor, so their share grows with context length.

Share of one forward pass's GPU kernel time (`cuda_gpu_kern_sum --filter-nvtx=forward`, `forward`-mode profiles):

| size | ctx | bs | fwd GPU (ms) | GEMM | elementwise | reduction | copy/index |
|---|--:|--:|--:|--:|--:|--:|--:|
| small | 256 | 4 | 30.0 | 65.4% | 28.2% | 3.9% | 2.5% |
| small | 512 | 4 | 70.4 | 58.7% | 34.4% | 4.8% | 2.2% |
| small | 1024 | 4 | 182.9 | 51.9% | 40.2% | 6.3% | 1.6% |
| medium | 256 | 4 | 100.2 | 72.1% | 22.9% | 2.8% | 2.2% |
| medium | 512 | 4 | 227.7 | 67.0% | 27.5% | 3.7% | 1.8% |
| medium | 1024 | 2 | 280.8 | 59.0% | 34.2% | 5.2% | 1.5% |

Where the non-GEMM time goes, small ctx 512 bs 4 (one forward pass, 70.4 ms of GPU time). The attention rows come from the NVTX ranges (`nvtx_kern_sum`). The other rows are mapped from kernel name and call count (12 layers):

| source | kernels (calls) | ms | % of fwd |
|---|---|--:|--:|
| attention softmax | strided `add` (subtract max, 12), `exp` (12), `Div` (12), `reduce Max` (12), `reduce sum` (12) | 10.15 | 14.4% |
| attention scale + causal mask | `Mul` by scalar (12), `where` (12) | 4.51 | 6.4% |
| SwiGLU | vectorized `Mul` (24: x·σ(x) and ·W3x), `sigmoid` (12) | 5.08 | 7.2% |
| RoPE + RMSNorm | strided `Mul` (146 = 96 RoPE + 50 RMSNorm), `CatArrayBatchedCopy` (24, RoPE concat), `pow` / `mean` / `rsqrt` (25 each) | 5.27 | 7.5% |
| residual + RoPE adds | vectorized `add` (72 = 24 residual + 48 RoPE) | 1.63 | 2.3% |
| head reshape | `direct_copy` (24, `rearrange(...).contiguous()`) | 0.67 | 1.0% |
| cross-entropy (on B×T×V logits) | one `max`, `sum`, `exp`, `log`, `neg`, `gather` each, 2× `add` | 1.70 | 2.4% |

At ctx 1024, the softmax (39.6 ms, 21.7%) and scale+mask (19.0 ms, 10.4%) kernels each take more GPU time than the attention GEMMs themselves (QKᵀ + PV: 20.9 ms, 11.4%). See (e).

Raw data: `results/data/2_1_4_nsys_profile/*_forward_cuda_gpu_kern_sum_nvtx=forward.csv`, `*_forward_nvtx_kern_sum.csv`.
