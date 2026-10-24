# 2.1.3 benchmarking_script — commands

(a) is the script itself: `cs336_systems/benchmark.py`. All runs use the defaults batch 4, context 512, fp32,
10 timed steps; each run is a fresh process and appends one JSON line to `--output`.

On the A10 (24 GB) a full training step for `large` doesn't fit (params + grads + 2 AdamW states ≈ 16 B/param ≈
15.5 GB before activations), so `large` runs with `--mode forward_backward`; `xl` and `10B` don't fit at all and are
disabled in `MODEL_SIZES`. Small and medium run with `--mode train`.

Smoke test: `uv run python -m cs336_systems.benchmark --size small --warmup-steps 1 --steps 2`.

## (b) 5 warm-up steps

```bash
rm -f results/data/2_1_3_benchmarking_script_b.jsonl
for cfg in small:train medium:train large:forward_backward; do
  IFS=: read size mode <<< "$cfg"
  uv run python -m cs336_systems.benchmark --size $size --mode $mode --warmup-steps 5 --steps 10 \
    --output results/data/2_1_3_benchmarking_script_b.jsonl
done 2>&1 | tee results/data/2_1_3_benchmarking_script_b.log
```

## (c) 0 / 1 / 2 / 5 warm-up steps

Warm-up count is the outer loop, so the runs are back-to-back in that order (the w=5 rows are a re-run, not
copied from (b)):

```bash
rm -f results/data/2_1_3_benchmarking_script_c.jsonl
for w in 0 1 2 5; do
  for cfg in small:train medium:train large:forward_backward; do
    IFS=: read size mode <<< "$cfg"
    uv run python -m cs336_systems.benchmark --size $size --mode $mode --warmup-steps $w --steps 10 \
      --output results/data/2_1_3_benchmarking_script_c.jsonl
  done
done 2>&1 | tee results/data/2_1_3_benchmarking_script_c.log
```

The writeup's "1st step" and "mean of steps 2–10" columns come from `timings.total.times_ms` (and the per-phase
split from `timings.{forward,backward,optimizer}.times_ms`) in the JSONL.

`*.log` files are gitignored, so only the `.jsonl` files are committed.
