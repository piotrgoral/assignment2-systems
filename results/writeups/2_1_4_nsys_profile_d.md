# 2.1.4 nsys_profile (d)

**Answer.** Matrix multiplies take 53–73% of the GPU time in inference (forward only) but just 45–53% of a full training step (small ctx 512: 60.6% → 48.4%), while elementwise kernels grow from 22–40% to 41–47%. Two things cause this. First, the AdamW step contains no matmuls at all: for small it is ~1,800 small elementwise kernels per step (`Mul`, `add`, `Div`, `sqrt`, `pow`, fill), which take 38 ms (small) / 123 ms (medium), 6–29% of the step. Second, the backward pass has a somewhat lower GEMM share than the forward pass (47–64% vs 52–72%), because it adds gradient kernels for softmax, mask, RoPE, SwiGLU and RMSNorm plus gradient-accumulation adds. Copy/index kernels also grow (2% → 3–6%), from the embedding backward (sort + scatter-add) and extra copies.

Share of GPU kernel time per step (`cuda_gpu_kern_sum` over the 3 profiled steps ÷ 3). `forward` mode = forward + loss only; `train` = forward + backward + AdamW:

| size | ctx | bs | mode | GPU / step (ms) | GEMM | elementwise | reduction | copy/index |
|---|--:|--:|---|--:|--:|--:|--:|--:|
| small | 256 | 4 | forward | 31.8 | 66.3% | 27.2% | 3.9% | 2.6% |
| small | 256 | 4 | train | 135.8 | 44.6% | 47.2% | 2.1% | 6.1% |
| small | 512 | 4 | forward | 74.6 | 60.6% | 32.7% | 4.6% | 2.2% |
| small | 512 | 4 | train | 262.6 | 48.4% | 44.2% | 2.9% | 4.5% |
| small | 1024 | 4 | forward | 186.0 | 52.6% | 39.6% | 6.2% | 1.6% |
| small | 1024 | 4 | train | 602.5 | 45.7% | 47.0% | 4.0% | 3.3% |
| medium | 256 | 4 | forward | 104.0 | 72.8% | 22.3% | 2.8% | 2.2% |
| medium | 256 | 4 | train | 426.0 | 47.6% | 44.8% | 1.6% | 6.0% |
| medium | 512 | 4 | forward | 233.1 | 67.7% | 26.9% | 3.6% | 1.8% |
| medium | 512 | 4 | train | 794.0 | 52.6% | 40.8% | 2.3% | 4.2% |
| medium | 1024 | 2 | forward | 284.3 | 59.4% | 33.9% | 5.2% | 1.5% |
| medium | 1024 | 2 | train | 961.0 | 47.1% | 45.8% | 3.2% | 3.8% |

Training step split by phase. The forward and optimizer phases come from the kernels inside their NVTX ranges (`nvtx_kern_sum`). Backward = step total − forward − optimizer, because autograd launches the backward kernels from its own thread, outside any NVTX range:

| size | ctx | bs | forward ms | fwd GEMM | backward ms | bwd GEMM | optimizer ms | opt GEMM | step GEMM |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| small | 256 | 4 | 30.4 | 65.4% | 67.9 | 59.8% | 37.5 | 0.0% | 44.6% |
| small | 512 | 4 | 72.2 | 59.4% | 152.8 | 55.2% | 37.6 | 0.0% | 48.4% |
| small | 1024 | 4 | 184.2 | 52.2% | 380.8 | 47.0% | 37.5 | 0.0% | 45.7% |
| medium | 256 | 4 | 100.5 | 72.2% | 202.8 | 64.3% | 122.7 | 0.0% | 47.6% |
| medium | 512 | 4 | 232.3 | 67.6% | 439.1 | 59.4% | 122.6 | 0.0% | 52.6% |
| medium | 1024 | 2 | 281.9 | 59.2% | 556.3 | 51.4% | 122.8 | 0.0% | 47.1% |

The optimizer's share is largest when the activations are cheap: 28% of the step for small ctx 256 versus 6% for small ctx 1024, since its cost depends only on the parameter count. Its kernels per step for small (111 parameter tensors): Mul (AUnary): 666 calls, 13.47 ms; add: 444 calls, 12.21 ms; Div (Binary): 111 calls, 3.19 ms; pow_tensor_scalar: 111 calls, 2.32 ms; FillFunctor: 222 calls, 2.17 ms; sqrt: 111 calls, 2.16 ms; add (in-place): 111 calls, 2.05 ms.

Raw data: `results/data/2_1_4_nsys_profile/*_cuda_gpu_kern_sum.csv`, `*_train_nvtx_kern_sum.csv`.
