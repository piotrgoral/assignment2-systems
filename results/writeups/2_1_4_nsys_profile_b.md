# 2.1.4 nsys_profile (b)

**Answer.** In every configuration, the cuBLAS fp32 GEMM `ampere_sgemm_128x64_tn` takes the most GPU time in the forward pass: 41–68% of it (small ctx 512: 51.7%). It runs 85 times per forward pass for small and 169 times for medium. That is one call per linear layer: 7 per block (Q, K, V, output, W1, W2, W3) × 12 or 24 layers, plus the LM head. It is still the top kernel over forward + backward, but its share drops to 13–23% (small ctx 512: 16.7%). That's because the backward pass's GEMMs (computing dX = dY·W and dW = dYᵀ·X) use other transpose variants (`_nn`, `_nt`) and CUTLASS SIMT kernels (`cutlass_80_simt_sgemm_*`), so the backward GEMM time is split across many kernel names.

"Forward" columns: `cuda_gpu_kern_sum --filter-nvtx=forward` on the `forward`-mode profiles, i.e. the kernels of exactly one forward pass (the first `forward` range). "fwd+bwd" columns: all kernels of the `train` profiles minus those inside the `optimizer` range, divided by 3 steps.

| size | ctx | bs | top forward kernel | calls / forward | % fwd GPU | top fwd+bwd kernel | calls / step | % fwd+bwd GPU | 2nd fwd+bwd kernel | % |
|---|--:|--:|---|--:|--:|---|--:|--:|---|--:|
| small | 256 | 4 | `ampere_sgemm_128x64_tn` | 85 | 60.5% | `ampere_sgemm_128x64_tn` | 85 | 18.6% | `cutlass_80_simt_sgemm_128x64_8x5_nn_align1` | 10.1% |
| small | 512 | 4 | `ampere_sgemm_128x64_tn` | 85 | 51.7% | `ampere_sgemm_128x64_tn` | 85 | 16.7% | `ampere_sgemm_128x64_nn` | 16.1% |
| small | 1024 | 4 | `ampere_sgemm_128x64_tn` | 85 | 41.3% | `ampere_sgemm_128x64_tn` | 85 | 13.5% | `Mul (Binary)` | 10.4% |
| medium | 256 | 4 | `ampere_sgemm_128x64_tn` | 169 | 68.0% | `ampere_sgemm_128x64_tn` | 169 | 22.6% | `ampere_sgemm_128x64_nn` | 11.0% |
| medium | 512 | 4 | `ampere_sgemm_128x64_tn` | 169 | 60.5% | `ampere_sgemm_128x64_tn` | 169 | 21.1% | `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` | 13.3% |
| medium | 1024 | 2 | `ampere_sgemm_128x64_tn` | 169 | 49.2% | `ampere_sgemm_128x64_tn` | 169 | 16.6% | `cutlass_80_simt_sgemm_256x128_8x4_nn_align1` | 10.5% |

The other GEMMs in one forward pass are the two attention batched matmuls: `ampere_sgemm_128x128_tn` for QKᵀ and `ampere_sgemm_128x128_nn` for PV, each called once per layer. Together they take only 6–11% of the forward pass, growing with context length (small ctx 512: 3.0% + 4.0%).

Raw data: `results/data/2_1_4_nsys_profile/*_cuda_gpu_kern_sum_nvtx=forward.csv`, `*_train_cuda_gpu_kern_sum.csv`, `*_train_nvtx_kern_sum.csv`.
