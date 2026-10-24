# 4.2 torch_compile — commands

## (a) Compiled vs eager attention

Same grid as 4.1.1 (batch 8, single head, d_model ∈ {16, 32, 64, 128} × seq_len ∈ {256, …, 16384}, 10 warm-up + 100
timed forward and 100 timed backward passes, no mask). Eager and compiled run interleaved, config by config, in one
process. Compiled attention is `torch.compile(scaled_dot_product_attention)`, rebuilt after `torch._dynamo.reset()`
for every config. The cuBLAS and TF32 warnings are filtered out of the log:

```bash
rm -f results/data/4_2_torch_compile_a.jsonl
uv run python -m cs336_systems.attention_benchmark --impls eager compiled --output results/data/4_2_torch_compile_a.jsonl 2>&1 \
  | grep -v -i "cublas\|run_backward\|TensorFloat32\|warnings.warn" | tee results/data/4_2_torch_compile_a.log
```

The seq_len 16384 rows were re-run (same command plus `--seq-lens 16384`) after a fix to `bench_config`. Before
the fix, an OOM anywhere in warm-up (which also runs backward) was reported as `oom_forward`. Compiled attention
actually OOMs in backward.

Smoke test plus a recompile check (prints nothing if no config recompiles):
`TORCH_LOGS=recompiles uv run python -m cs336_systems.attention_benchmark --impls eager compiled --d-models 16 32 --seq-lens 256 1024 --steps 5`.

Correctness: compiled vs eager output, max abs diff ≈ 1e-7 (≈ 5e-7 with a causal mask) at B=8, N=1024, d=64.

## (b) Compiled vs vanilla Transformer

ctx 512, batch 4, 5 warm-up + 10 timed steps, the same as 2.1.3 (b). `--compile` wraps only the model in
`torch.compile`; the loss and AdamW stay eager. Large uses `forward_backward`, because a full training step OOMs on the A10.
`large train --compile` was tried as well and also OOMs (19.8 GiB allocated), so it is not in the jsonl.

```bash
out=results/data/4_2_torch_compile_b.jsonl; rm -f $out
for cfg in "small forward" "small train" "medium forward" "medium train" "large forward" "large forward_backward"; do
  set -- $cfg
  for c in "" "--compile"; do
    echo "=== size=$1 mode=$2 $c"
    uv run python -m cs336_systems.benchmark --size $1 --mode $2 $c --warmup-steps 5 --steps 10 --output $out 2>&1 \
      | grep -v -i "cublas\|run_backward\|TensorFloat32\|warnings.warn"
  done
done 2>&1 | tee results/data/4_2_torch_compile_b.log
```

## Tables

The writeup tables are generated from the jsonl files with:

```python
import json

def load(p):
    return [json.loads(l) for l in open(p)]

# (a)
rows = {(r["d_model"], r["seq_len"], r["impl"]): r for r in load("results/data/4_2_torch_compile_a.jsonl")}
print("| d_model | seq_len | fwd eager | fwd compiled | fwd speedup | bwd eager | bwd compiled | bwd speedup | mem before bwd (MiB) | peak eager / compiled (MiB) |")
print("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
for d, n, _ in sorted(k for k in rows if k[2] == "eager"):
    e, c = rows[d, n, "eager"], rows[d, n, "compiled"]
    if e["status"] != "ok":
        print(f"| {d} | {n} | {e['status']} | {c['status']} | – | – | – | – | – | – |")
        continue
    fe, fc = e["forward"]["mean_ms"], c["forward"]["mean_ms"]
    be, bc = e["backward"]["mean_ms"], c["backward"]["mean_ms"]
    print(f"| {d} | {n} | {fe:.2f} | {fc:.2f} | {fe/fc:.2f}× | {be:.2f} | {bc:.2f} | {be/bc:.2f}× "
          f"| {e['mem_before_backward_bytes']/2**20:.1f} | {e['peak_mem_bytes']/2**20:.0f} / {c['peak_mem_bytes']/2**20:.0f} |")

# (b)
print()
print("| size | mode | phase | vanilla (ms) | compiled (ms) | speedup |")
print("|---|---|---|--:|--:|--:|")
rs = load("results/data/4_2_torch_compile_b.jsonl")
by = {(r["size"], r["mode"], r["compiled"]): r for r in rs}
for (size, mode, comp), v in by.items():
    if comp:
        continue
    c = by[size, mode, True]
    for ph, t in v["timings"].items():
        a, b = t["mean_ms"], c["timings"][ph]["mean_ms"]
        print(f"| {size} | {mode} | {ph} | {a:.1f} ± {t['std_ms']:.1f} | {b:.1f} ± {c['timings'][ph]['std_ms']:.1f} | {a/b:.2f}× |")
print()
for (size, mode, comp), r in by.items():
    if comp:
        print(f"first step (compile) {size} {mode}: {r['first_step_s']:.1f} s (vanilla {by[size, mode, False]['first_step_s']:.1f} s)")
```
