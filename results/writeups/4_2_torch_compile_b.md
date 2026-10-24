# 4.2 torch_compile (b)

The whole `BasicsTransformerLM` is wrapped in `torch.compile`; the loss and the AdamW step stay eager. fp32, ctx 512,
batch 4, 5 warm-up + 10 timed steps on an A10, the same setup as 2.1.3 (b). Values are mean ± std over the 10 steps.
"forward" mode is a forward pass only (with autograd on). "train" is forward + backward + AdamW step. Large uses
forward + backward, because a full training step does not fit in memory, with or without compile (compiled:
OOM at 19.8 GiB allocated).

| size | mode | phase | vanilla (ms) | compiled (ms) | speedup |
|---|---|---|--:|--:|--:|
| small | forward | total | 78.5 ± 2.0 | 60.7 ± 0.9 | 1.29× |
| small | forward | forward | 78.4 ± 2.0 | 60.7 ± 0.9 | 1.29× |
| small | train | total | 269.4 ± 1.5 | 211.4 ± 1.4 | 1.27× |
| small | train | forward | 73.9 ± 0.8 | 55.8 ± 0.8 | 1.33× |
| small | train | backward | 155.9 ± 0.9 | 116.1 ± 0.9 | 1.34× |
| small | train | optimizer | 38.8 ± 0.1 | 39.1 ± 0.0 | 0.99× |
| medium | forward | total | 238.8 ± 2.1 | 194.6 ± 4.9 | 1.23× |
| medium | forward | forward | 238.8 ± 2.1 | 194.6 ± 4.9 | 1.23× |
| medium | train | total | 800.3 ± 1.1 | 647.6 ± 1.6 | 1.24× |
| medium | train | forward | 232.7 ± 1.0 | 187.1 ± 0.6 | 1.24× |
| medium | train | backward | 441.8 ± 0.8 | 335.3 ± 1.1 | 1.32× |
| medium | train | optimizer | 124.4 ± 0.1 | 124.5 ± 0.1 | 1.00× |
| large | forward | total | 499.4 ± 2.0 | 416.9 ± 2.1 | 1.20× |
| large | forward | forward | 499.4 ± 2.0 | 416.9 ± 2.1 | 1.20× |
| large | forward_backward | total | 1466.9 ± 3.9 | 1185.9 ± 4.5 | 1.24× |
| large | forward_backward | forward | 500.0 ± 2.1 | 423.0 ± 1.2 | 1.18× |
| large | forward_backward | backward | 963.0 ± 2.0 | 760.9 ± 4.0 | 1.27× |

first step (compile) small forward: 16.8 s (vanilla 0.3 s)

Compilation cost, i.e. time of the first warm-up step: 17 / 31 / 48 s for small / medium / large in forward mode, and
19 / 36 / 58 s when the backward graph is compiled too. The uncompiled first step takes 0.3–1.7 s.

**Answer.** Compiling the model speeds up the forward pass by 1.2–1.3×: 78 → 61 ms (small), 239 → 195 ms
(medium), 499 → 417 ms (large). The gain shrinks as the model grows. Inductor fuses the elementwise and reduction ops
(RMSNorm, SwiGLU, RoPE, the attention softmax, residual adds) but leaves the linear-layer matmuls to cuBLAS, and those
matmuls make up a larger share of the work in bigger models. The backward pass gains a little more (1.27–1.34×), because it
has more elementwise ops to fuse (the backward passes of the activation, norm and softmax). A full training step is
1.27× faster for small (269 → 211 ms) and 1.24× for medium (800 → 648 ms). The optimizer step is not compiled and is
unchanged (39 / 124 ms), so it slightly dilutes the overall speedup. The one-time cost is 20–60 s of compilation, which
pays for itself after about 200–330 steps (≈ 330 for small, ≈ 235 for medium, ≈ 205 for large forward + backward).

Raw data: `results/data/4_2_torch_compile_b.jsonl` (log: `results/data/4_2_torch_compile_b.log`); commands in
`results/notes/4_2_torch_compile.md`.
