# 4.1.1 pytorch_attention — commands

Full grid (batch 8, single head, d_model ∈ {16, 32, 64, 128} × seq_len ∈ {256, 1024, 4096, 8192, 16384},
10 warm-up + 100 timed forward and 100 timed backward passes, no mask). The cuBLAS "no current CUDA context"
warning is harmless and is filtered out of the log:

```bash
rm -f results/data/4_1_pytorch_attention.jsonl
uv run python -m cs336_systems.attention_benchmark --output results/data/4_1_pytorch_attention.jsonl 2>&1 \
  | grep -v -i "cublas\|run_backward" | tee results/data/4_1_pytorch_attention.log
```

Smoke test: `uv run python -m cs336_systems.attention_benchmark --d-models 16 --seq-lens 256 --steps 5`.

Listing the tensors autograd saves (used for the memory accounting in the writeup):

```python
seen = {}
def pack(t):
    seen[t.data_ptr()] = (tuple(t.shape), t.dtype); return t
with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    out = scaled_dot_product_attention(Q, K, V)
```
