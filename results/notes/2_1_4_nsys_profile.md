# 2.1.4 nsys_profile — commands

Profiles land in `results/data/2_1_4_nsys_profile/` as `<size>_ctx<ctx>_bs<batch>_<mode>.nsys-rep`, with
kernel / NVTX summaries next to them as `.csv`.

## 0. One-time setup

Don't use the Lambda `nsight-systems` apt package (2024.6.2, `/usr/bin/nsys`): its `QdstrmImporter` links
against a newer `libssh` than Ubuntu 22.04 has (`LIBSSH_4_9_0 not found`), so every run leaves only a raw
`.qdstrm` and no `.nsys-rep`.

Instead, unpack NVIDIA's own build (2026.3.2) into the home directory. The `.deb` comes from NVIDIA's CUDA
repo (no login needed) and is only extracted, not installed: no apt / system changes, and the Lambda packages
can stay installed. It bundles its own libraries.

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2026.3.2_2026.3.2.476-1_amd64.deb
dpkg-deb -x nsight-systems-2026.3.2_2026.3.2.476-1_amd64.deb ~/nsight-systems
```

Put it ahead of the Lambda `nsys` on `PATH` (needed in every shell that runs the commands below):

```bash
echo 'export PATH=~/nsight-systems/opt/nvidia/nsight-systems/2026.3.2/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
hash -r && which nsys && nsys --version   # ~/nsight-systems/.../bin/nsys, version 2026.3.2
```

## 1. Grid

Two sizes × three context lengths, for `forward` and `train` (forward + backward + AdamW step).
On the A10 (24 GB), training at batch 4 OOMs for `small` at ctx 2048 and for `medium` at ctx 1024,
so `medium` ctx 1024 runs at batch 2.

| size   | ctx 256 | ctx 512 | ctx 1024 |
|--------|---------|---------|----------|
| small  | bs 4    | bs 4    | bs 4     |
| medium | bs 4    | bs 4    | bs 2     |

## 2. Profile

`NSYS_NVTX_PROFILER_REGISTER_ONLY=0` is required: PyTorch's NVTX strings aren't registered, so without it
`--nvtx-capture=measure` never fires. The capture range skips the 5 warmup steps; 3 measured steps are recorded.
`osrt` tracing is left out (it bloated the raw trace to >2 GB for the smallest config), and CPU sampling /
context-switch tracing are disabled since this VM doesn't support them anyway.

```bash
cd /home/ubuntu/assignment2-systems
OUT=results/data/2_1_4_nsys_profile
mkdir -p $OUT

for cfg in small:256:4 small:512:4 small:1024:4 medium:256:4 medium:512:4 medium:1024:2; do
  IFS=: read size ctx bs <<< "$cfg"
  for mode in forward train; do
    NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile \
      --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --capture-range=nvtx --nvtx-capture=measure --capture-range-end=stop \
      --force-overwrite=true \
      -o $OUT/${size}_ctx${ctx}_bs${bs}_${mode} \
      python -m cs336_systems.benchmark --size $size --context-length $ctx --batch-size $bs \
        --mode $mode --warmup-steps 5 --steps 3 --annotate-attention
  done
done
```

Single run with the richer options from the handout (slower; backtraces have large overhead):

```bash
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx \
  --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 \
  --capture-range=nvtx --nvtx-capture=measure --capture-range-end=stop \
  --force-overwrite=true -o results/data/2_1_4_nsys_profile/small_ctx512_bs4_train_detailed \
  python -m cs336_systems.benchmark --size small --context-length 512 --mode train \
    --warmup-steps 5 --steps 3 --annotate-attention
```

## 3. Extract stats

Each report is written as `<base>_<report>.csv` next to the `.nsys-rep` (nsys also leaves a `<base>.sqlite`).

| report | file suffix | what it gives | questions |
|---|---|---|---|
| `nvtx_sum` | `_nvtx_sum.csv` | time of each NVTX range (`forward`, `backward`, `optimizer`, `step_*`, attention parts) | (a) |
| `cuda_gpu_kern_sum` | `_cuda_gpu_kern_sum.csv` | GPU kernels over all 3 measured steps (divide counts by 3) | (b), (d) |
| `cuda_gpu_kern_sum --filter-nvtx=forward` | `_cuda_gpu_kern_sum_nvtx=forward.csv` | kernels of the **first** `forward` range only, i.e. exactly one forward pass | (b), (c) |
| `nvtx_kern_sum` | `_nvtx_kern_sum.csv` | GPU kernel time per NVTX range, summed over all instances (all layers / steps) | (d), (e) |

```bash
OUT=results/data/2_1_4_nsys_profile
for R in $OUT/*.nsys-rep; do
  base=${R%.nsys-rep}
  # --force-export: re-create <base>.sqlite if the .nsys-rep was re-profiled (a stale one makes nsys stats fail)
  nsys stats -q --force-export=true -r nvtx_sum,cuda_gpu_kern_sum,nvtx_kern_sum -f csv -o $base --force-overwrite=true $R
  nsys stats -q -r cuda_gpu_kern_sum --filter-nvtx=forward -f csv -o $base --force-overwrite=true $R
done
```

For (e), compare the `computing softmax` rows against the `computing attention scores` + `final matmul` rows in
`_nvtx_kern_sum.csv` (filter with `grep -E '^"?:(computing|final)'`).

Quick look at one report in the terminal:

```bash
R=results/data/2_1_4_nsys_profile/small_ctx512_bs4_train.nsys-rep
nsys stats -r nvtx_sum $R
nsys stats -r cuda_gpu_kern_sum --filter-nvtx=forward $R
```

Compare (a) against the `timeit` numbers in `results/data/2_1_3_benchmarking_script_b.jsonl`.

## 4. GUI (optional)

Copy `.nsys-rep` files to a local machine and open them in the Nsight Systems desktop app:

```bash
scp 'ubuntu@<host>:assignment2-systems/results/data/2_1_4_nsys_profile/*.nsys-rep' .
```
