# 4.2.2 flash_forward — commands

## (a) Pure-PyTorch FlashAttention-2 forward

Implementation: `FlashAttentionPytorch` in `cs336_systems/flash_attention.py`, returned by
`tests/adapters.py::get_flashattention_autograd_function_pytorch`. It follows Algorithm 1 with tiles of 16 × 16
(the minimum allowed), runs over all leading batch dims at once and loops over tiles in Python. The online softmax
state (m, l, O_i) is kept in fp32. It saves `(L, Q, K, V, O)` and returns O. `is_causal` is stored on ctx but not
used yet, and `backward` raises `NotImplementedError` until flash_backward is done.

The test (CPU, Q/K/V of shape (4, 128, 64), atol/rtol 1e-2 against `_attention_and_lse`) also checks that exactly one
saved tensor has shape (batch, n_queries), i.e. L:

```bash
uv run pytest -k test_flash_forward_pass_pytorch
```

Extra check with N_q ≠ N_k and two batch dims. Max abs diff vs the reference: O ≈ 3e-7, L ≈ 5e-7. The inputs need
`requires_grad=True`, otherwise there is no `grad_fn` to read the saved L from:

```python
import torch
from tests.test_attention import _attention_and_lse
from cs336_systems.flash_attention import FlashAttentionPytorch

q, k, v = [torch.randn(2, 3, n, 32, requires_grad=True) for n in (64, 256, 256)]
o = FlashAttentionPytorch.apply(q, k, v, False)
o_ref, l_ref = _attention_and_lse(q, k, v)
print((o - o_ref).abs().max().item(), (o.grad_fn.saved_tensors[0] - l_ref).abs().max().item())
```

## Extra: benchmarking the PyTorch forward

Writeup: `results/writeups/4_2_2_flash_forward_pytorch_benchmark.md`. `attention_benchmark.py` has a `flash_pytorch`
impl. It is forward-only (there is no backward yet), uses square tiles of `--flash-tile-size` and refuses `--causal`.
The grid is the same as 4.1.1, but with 2 warm-up + 10 timed passes, because a pass at seq_len 16384 takes ~2.7 s with
128 tiles. Total run time is about 5 min:

```bash
out=results/data/4_2_2_flash_forward_pytorch.jsonl; rm -f $out
uv run python -m cs336_systems.attention_benchmark --impls flash_pytorch --flash-tile-size 128 --warmup-steps 2 --steps 10 --output $out 2>&1 \
  | grep -v -i "cublas\|run_backward" | tee results/data/4_2_2_flash_forward_pytorch.log
# tile-size comparison at the test's 16 x 16 tiles (~0.68 s per pass at seq_len 1024)
uv run python -m cs336_systems.attention_benchmark --impls flash_pytorch --flash-tile-size 16 --d-models 64 --seq-lens 256 1024 --warmup-steps 1 --steps 3 --output $out 2>&1 \
  | grep -v -i "cublas\|run_backward" | tee -a results/data/4_2_2_flash_forward_pytorch.log
```

Smoke test: `uv run python -m cs336_systems.attention_benchmark --impls eager flash_pytorch --d-models 16 --seq-lens 256 --steps 3 --warmup-steps 1`.

There is no torch.compile run: Dynamo would unroll the Python tile loops into a (N/Bq)·(N/Bk)-iteration graph (see the writeup).

## (b) Triton FlashAttention-2 forward

Implementation: `flash_fwd_kernel` + `FlashAttentionTriton` in `cs336_systems/flash_attention.py`, returned by
`tests/adapters.py::get_flashattention_autograd_function_triton`. The kernel has the handout's signature and grid
`(T_q, batch_size)`. Each program loads its Q_i tile once and then runs a single loop over key tiles. It keeps
O_i, l, m in fp32, casts P̃ to V's dtype before `tl.dot(P̃, V_j, acc=alpha * O_i)`, and advances the K/V block
pointers at the end of the loop. O is cast to the output's element type on store, and L (fp32) is stored through a
1-D block pointer. The wrapper flattens the leading batch dims into one before the launch and reshapes O and L back
afterwards. Tiles are 16 × 16 for now (class attributes, not tuned yet). `is_causal` is saved on ctx but not applied
yet, so the `[True]` case of the test still fails until (c):

```bash
uv run pytest -k test_flash_forward_pass_triton   # [False] passes, [True] fails (no mask yet)
```

Extra check against `_attention_and_lse` (in fp32) and `FlashAttentionPytorch`, max abs diff:

| Q shape / K shape | dtype | O vs ref | L vs ref | O vs pytorch flash |
|---|---|---:|---:|---:|
| (2, 3, 64, 32) / (2, 3, 256, 32) | fp32 | 1.1e-3 | 1.7e-3 | 1.1e-3 |
| (8, 1024, 128) | fp32 | 7.9e-4 | 1.3e-3 | 7.9e-4 |
| (8, 1024, 64) | bf16 | 1.1e-3 | 1e-6 | 2.0e-3 |

The ~1e-3 error in fp32 is from `tl.dot` using TF32 by default. That is fine for the test's 1e-2 tolerance.
(Later the kernel switched to `input_precision="ieee"` for fp32, and the fp32 error dropped to ~1e-6. See
`4_2_flash_benchmarking.md`, "Fixes after review".)
Quick timing at (8, 4096, 64) fp32 with `triton.testing.do_bench`: Triton 3.5 ms vs eager
`scaled_dot_product_attention` 15.0 ms, with untuned tiles and no full benchmark yet.

## Extra: benchmarking the Triton forward

Writeup: `results/writeups/4_2_3_flash_forward_triton_benchmark.md`. `attention_benchmark.py` has a `flash_triton`
impl. Like `flash_pytorch`, it is forward-only, uses square tiles of `--flash-tile-size` and refuses `--causal`. A
tile size that needs more shared memory than the GPU has (Triton `OutOfResources`) is recorded as
`"status": "out_of_shared_memory"`. Same grid and step counts as 4.1.1 (10 warm-up + 100 timed), swept over four
tile sizes. Total run time is about 1 min:

```bash
out=results/data/4_2_3_flash_forward_triton.jsonl; rm -f $out
for t in 16 32 64 128; do
  uv run python -m cs336_systems.attention_benchmark --impls flash_triton --flash-tile-size $t --output $out 2>&1 \
    | grep -v -i "cublas\|run_backward"
done | tee results/data/4_2_3_flash_forward_triton.log
```

Smoke test: `uv run python -m cs336_systems.attention_benchmark --impls eager flash_triton --d-models 16 --seq-lens 256 --steps 3 --warmup-steps 1`.

## Next steps

Done: (c) causal masking (`is_causal: tl.constexpr`, -1e6 on masked scores, `ctx.is_causal`) and flash_backward
(compiled PyTorch, Eq. 13–19). Both are covered by `uv run pytest -k flash`. Benchmarks: `4_2_flash_benchmarking.md`.
