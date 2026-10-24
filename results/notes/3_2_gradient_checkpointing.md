# 3.2 gradient_checkpointing — commands

Script: `cs336_systems/gradient_checkpointing.py`. Like the handout's `four_blocks` example, one xl `TransformerBlock`
(d_model 2560, d_ff 10240, 16 heads, RoPE, `torch.compile(fullgraph=True)`) is applied N times. One step =
forward + `y.sum().backward()` on a fresh `x` of shape (4, 2048, 2560) with `requires_grad=True`. The full xl LM can't be
used on the A10: ~3.4B params → ~27 GB of fp32 params + grads. With one shared block, params + grads are 822 MiB
(`base_mib`), so the peak is almost all activations.

Measured per run (fresh process per config): `peak_mib` = `torch.cuda.max_memory_allocated()` over the measured
steps (after 1 warm-up step that also compiles); `activation_peak_mib` = peak − memory allocated before the step;
`saved_outside_checkpoints_mib` = bytes the handout's pack hook sees during the forward pass (= the checkpoints, since the
non-reentrant checkpoint suppresses everything saved inside it).

Correctness (gradients of x and of all params vs. eager `none`, small config, eager and compiled; max rel. err ≤ 3e-7):

```bash
uv run python -m cs336_systems.gradient_checkpointing --check
```

## (b) one level of checkpointing, xl, batch 4, ctx 2048, N = 32

```bash
out=results/data/3_2_gradient_checkpointing_b.jsonl; rm -f $out
for s in "none" "sublayer" "uniform --block-size 1" "uniform --block-size 2" "uniform --block-size 4" "uniform --block-size 8"; do
  echo "=== $s"
  uv run python -m cs336_systems.gradient_checkpointing --strategy $s --output $out 2>&1 | grep -v -i "cublas\|run_backward\|TensorFloat32\|warnings.warn"
done 2>&1 | tee results/data/3_2_gradient_checkpointing_b.log
```

Memory snapshot of the winner (open in https://pytorch.org/memory_viz):

```bash
uv run python -m cs336_systems.gradient_checkpointing --strategy uniform --block-size 1 --steps 1 \
  --memory-snapshot results/data/3_2_gradient_checkpointing_b_snapshot.pickle
```

## (a) support: nested (recursive) vs. one block per checkpoint, scaling in N

```bash
for n in 8 16 32 64; do for s in "recursive" "uniform --block-size 1"; do
  uv run python -m cs336_systems.gradient_checkpointing --strategy $s --num-blocks $n --steps 1 \
    --output results/data/3_2_gradient_checkpointing_a.jsonl 2>&1 | grep -v -i "cublas\|run_backward\|TensorFloat32\|warnings.warn" | sed "s/^/N=$n /"
done; done | tee results/data/3_2_gradient_checkpointing_a.log
```
