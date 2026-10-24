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

## Next steps

- (b) Triton forward kernel following Algorithm 1: `get_flashattention_autograd_function_triton`, test with
  `uv run pytest -k test_flash_forward_pass_triton`. Use `FlashAttentionPytorch` as the reference when debugging,
  tile by tile.
- (c) Causal masking in the Triton kernel: an `is_causal: tl.constexpr` parameter, add -1e6 to the masked entries of
  S_ij, keep `ctx.is_causal = is_causal`. Same test, parametrized with `is_causal=True`.
- flash_backward: implement `backward` (PyTorch, can be `torch.compile`d) from the saved `L, Q, K, V, O` with
  recomputation (Eq. 13–19, including the D vector). Test with
  `uv run pytest -k test_flash_backward`.
