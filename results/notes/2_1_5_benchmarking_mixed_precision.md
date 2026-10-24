# 2.1.5 benchmarking_mixed_precision — commands

(a)/(b) use `cs336_systems/toy_mixed_precision.py`: the handout's ToyModel with fp32 params, forward + loss under
`torch.autocast`, backward outside it; layer outputs are captured with forward hooks. Run with fp16 (asked in (a))
and with bf16 (evidence for (b)).

```bash
rm -f results/data/2_1_5_benchmarking_mixed_precision_a.jsonl
for dt in fp16 bf16; do
  uv run python -m cs336_systems.toy_mixed_precision --dtype $dt \
    --output results/data/2_1_5_benchmarking_mixed_precision_a.jsonl
done 2>&1 | tee results/data/2_1_5_benchmarking_mixed_precision_a.log
```

(c) uses `cs336_systems/benchmark.py --mixed-precision bf16`, which wraps forward + loss in
`torch.autocast(dtype=torch.bfloat16)` (`nullcontext` otherwise); backward stays outside autocast. All sizes run with
`--mode forward_backward` (the question asks for forward and backward, and `large` doesn't fit a full train step on the
A10), batch 4, context 512, 5 warm-up / 10 timed steps, a fresh process per run; fp32 and bf16 run back-to-back per size.

```bash
rm -f results/data/2_1_5_benchmarking_mixed_precision_c.jsonl
for size in small medium large; do
  for mp in none bf16; do
    uv run python -m cs336_systems.benchmark --size $size --mode forward_backward --mixed-precision $mp \
      --warmup-steps 5 --steps 10 --output results/data/2_1_5_benchmarking_mixed_precision_c.jsonl
  done
done 2>&1 | tee results/data/2_1_5_benchmarking_mixed_precision_c.log
```

Sanity check: at init the small model's loss is 9.2808 in fp32 and 9.2807 under bf16 autocast (loss dtype fp32).

`*.log` files are gitignored, so only the `.jsonl` files are committed.
