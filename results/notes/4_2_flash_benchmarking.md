# 4.2 flash_benchmarking — commands

## Kernel changes before benchmarking

In `cs336_systems/flash_attention.py`:

- **Causal tile skipping.** With `is_causal`, `flash_fwd_kernel` stops its key loop at
  `cdiv((query_tile_index + 1) * Bq, Bk)`. Later key tiles are fully masked, and with the additive -1e6 they would
  add exactly 0, so the output is unchanged. This does about half the work of the full loop.
- **Autotuned tiles.** `flash_fwd_kernel_autotuned` is `triton.autotune` over (Bq, Bk) ∈ {16², 32², 32×64,
  64×32, 64², 128×32, 128×64, 128²} × `num_warps` ∈ {4, 8} × `num_stages` ∈ {2, 3}. It is keyed on
  `N_QUERIES, N_KEYS, D, is_causal`, and Triton 3.6 also adds the tensors' dtypes to the key. Tiles that do not
  divide the sequence lengths are pruned, and configs that need too much shared memory get an inf time.
  `FlashAttentionTriton.Q_TILE_SIZE / K_TILE_SIZE` now default to `None` (autotune). Setting them forces fixed tiles,
  which `attention_benchmark.py --flash-tile-size` still does.

Checks: `uv run pytest -k flash` (6 passed). Causal, N = 1024, d = 64: max abs diff vs `_attention_and_lse` is 2.2e-3
(fp32 / TF32) and 4.3e-3 (bf16). These are the same values as with fixed 16 × 16 tiles.

## Benchmark

Script: `cs336_systems/flash_benchmark.py`. It uses `triton.testing.do_bench` medians, batch 1, causal, seq_len
2^7 … 2^16 × d_model {16, 32, 64, 128} × {bf16, fp32}, with `pytorch` (cs336_basics `scaled_dot_product_attention` +
tril mask) vs `flash` (`FlashAttentionTriton`). One untimed fwd+bwd runs first (autotuning, torch.compile). OOMs are
recorded per stage.

```bash
out=results/data/4_2_flash_benchmarking.jsonl; rm -f $out
uv run python -m cs336_systems.flash_benchmark --output $out 2>&1 | grep -v -i "warn\|run_backward" > results/data/4_2_flash_benchmarking.log
```

Smoke test (~1 min): `uv run python -m cs336_systems.flash_benchmark --seq-lens 128 1024 --d-models 16 --dtypes bfloat16`.

Writeup: `results/writeups/4_2_flash_benchmarking.md`.

The first full run warmed up with forward + backward together, so every N = 65536 row came out `oom_warmup`, even
though the flash forward fits. The script now warms up and times the forward before any backward. seq_len 32768 and
65536 were re-run with it and replaced in the jsonl. The other rows are unaffected, because nothing OOMed there.

```bash
uv run python -m cs336_systems.flash_benchmark --seq-lens 32768 65536 --output <tmp>.jsonl
```

Causal-skip check (forward, N = 16384): the non-causal (full loop) time divided by the causal time is 1.48 / 1.36
(bf16 d = 16 / 128) and 1.37 / 1.52 (fp32).
