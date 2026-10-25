Benchmarking (benchmark() in the lecture)
  - Every timed loop has warm-up steps, and torch.compile and Triton autotuning happen during warm-up. benchmark.py records the first step separately as first_step_s.
  - The GPU is synchronized before each timer stops: benchmark.py:155, benchmark.py:226, attention_benchmark.py:128 and :152, gradient_checkpointing.py:488.
  - Each run has multiple trials, and you report mean, std, min and max, which is more than the lecture's plain mean.
  - flash_benchmark.py uses triton.testing.do_bench. It times with CUDA events, which the lecture recommends to keep CPU overhead out of the measurement, and it reports
    medians.

  Profiling
  - NVTX ranges around warmup, measure and each step and phase cover what the lecture says about using nsight to see where time goes.

  The Triton kernel (flash_attention.py)
  - It is written the way the lecture describes Triton: one program per tile. Each program loads a Q tile into on-chip memory, loops over K/V tiles, and writes O and L back to
    HBM once. That is the tiling and fusion pattern from matmul_relu_kernel.
  - Loads and stores have boundary checks (boundary_check), like mask=offsets < num_elements in the lecture.
  - Strides are passed in explicitly rather than assuming a layout, as in triton_matmul_relu.
  - Accumulators stay in fp32 (acc=O_i * alpha, like acc = tl.zeros(..., tl.float32) in the lecture).
  - Tiles are 2-D block pointers with order=(1, 0), so loads along the head dimension are coalesced.
  - The softmax subtracts the row max, as triton_softmax_kernel does.
  - With causal masking, the key loop stops at the diagonal, so fully masked tiles are never loaded.

  Issues

  1. Bug: padded keys aren't masked out, so the result is wrong (flash_attention.py:181-187, :211)

  The lecture's softmax kernel loads out-of-range columns with other=float("-inf") so they contribute nothing. Your kernel pads K with zeros, so a padded key gets a score of 0
  instead of −inf. In the non-causal case, exp(0 - m) is then added to l_i and the normalization comes out wrong.

  The fixed-tile path asserts that sequence lengths divide the tile size, so it is safe. The autotuned path is not: when no tile size divides the length, _prune_tile_configs
  falls back to 16×16 tiles without checking. I confirmed this on the GPU against PyTorch's scaled_dot_product_attention:

  n=100 causal=False  max err 0.047   ← wrong
  n=100 causal=True   max err 0.002   (the causal mask happens to hide the padded keys)
  n=128 causal=False  max err 0.002

  The fix is to mask padded keys the way the lecture does. This would also let you delete the divisibility pruning:
  k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
  S_ij = tl.where(k_idx[None, :] < N_KEYS, S_ij, float("-inf"))
  Move the k_idx line out of the if is_causal: block so both masks can use it.

  2. Small: fp32 inputs are multiplied in TF32 (flash_attention.py:184, :192)

  By default, tl.dot on fp32 inputs uses TF32, which is why the error is about 2e-3 even in the passing cases above. The lecture's correctness checks use atol=1e-6. If you
  want true fp32 for the fp32 benchmarks, pass input_precision="ieee"; otherwise, say in your write-up that fp32 there means TF32.

  3. Design note: the backward pass builds the full N×N matrix (flash_attention.py:39-47)
  want true fp32 for the fp32 benchmarks, pass input_precision="ieee"; otherwise, say in your write-up that fp32 there means TF32.

  want true fp32 for the fp32 benchmarks, pass input_precision="ieee"; otherwise, say in your write-up that fp32 there means TF32.

  3. Design note: the backward pass builds the full N×N matrix (flash_attention.py:39-47)

  The backward recomputes S and P for all queries and keys at once, so it isn't tiled. The handout allows a torch.compile backward, but in your flash_benchmark results the
  backward memory still grows as O(N²), and an OOM at long sequences will come from the backward, not the forward.

  4. Performance note: too few thread blocks at batch size 1 (flash_benchmark.py)

  The grid is (number of query tiles, batch). At batch 1 with seq_len 128 and Q tile 64, only 2 programs launch on a GPU with around 100 or more SMs, so most SMs sit idle. This
  is the lecture's wave-quantization / block-occupancy point. It isn't a bug, but it explains why flash attention doesn't beat PyTorch at short sequences.

  5. What gets timed
  - benchmark.py:155: PhaseTimer synchronizes after every phase, so total includes two or three extra syncs. That removes the CPU/GPU overlap between forward, backward and the
    optimizer step, so total is slightly higher than an unsynchronized step would take. Note it, or time total in a separate run without phase syncs.
  - gradient_checkpointing.py:487: the timed steps run under the SavedTensorCounter hook, so step_mean_ms includes its Python overhead. Time without the hook if you report step
    times.

  6. Out-of-date docstring (attention_benchmark.py:17-24, :45, :186)

  The docstring still says the flash implementations have no backward and ignore masks, and FORWARD_ONLY_IMPLS and the --causal guard enforce that. Both classes now support
  backward and is_causal, so the 4.1 benchmark is skipping things that would work.

  mixed_precision_accumulation.py and toy_mixed_precision.py cover topics this lecture doesn't, so I didn't review them against it.