"""Run commands from this repo on a Modal GPU box.

Examples:
    modal run modal_runner.py                                   # pytest -q
    modal run modal_runner.py --command "pytest -q tests/test_attention.py"
    modal run modal_runner.py --command "python -m cs336_systems.benchmark --size small --output results/data/foo.jsonl"
    modal run modal_runner.py --gpu H100 --command "python -m cs336_systems.flash_benchmark"

Anything the command writes under results/ is copied back into the local results/ directory.
"""

import io
import tarfile
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent
REMOTE = "/workspace"
VENV = "/opt/cs336-venv"

app = modal.App("cs336-systems")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("uv")
    .env({"UV_PROJECT_ENVIRONMENT": VENV, "UV_LINK_MODE": "copy"})
    # Install dependencies in a cached layer so torch etc. are not reinstalled on every run;
    # this layer is rebuilt only when pyproject.toml, uv.lock or cs336-basics change.
    .add_local_file(str(REPO / "pyproject.toml"), f"{REMOTE}/pyproject.toml", copy=True)
    .add_local_file(str(REPO / "uv.lock"), f"{REMOTE}/uv.lock", copy=True)
    .add_local_dir(str(REPO / "cs336-basics"), f"{REMOTE}/cs336-basics", copy=True, ignore=["**/__pycache__", "*.pyc"])
    .run_commands(f"cd {REMOTE} && uv sync --frozen --no-install-project")
    # The rest of the repo is mounted at container start, so code edits don't trigger an image rebuild.
    .add_local_dir(
        str(REPO),
        remote_path=REMOTE,
        ignore=[
            ".git",
            ".venv",
            "**/__pycache__",
            ".pytest_cache",
            "*.pyc",
            "*.pdf",
            "cs336_assignment2_systems.txt",
            "pyproject.toml",
            "uv.lock",
            "cs336-basics",
        ],
    )
)


@app.function(image=image, gpu="A10:2", cpu=4, memory=16384, timeout=1800)
def run_command(command: str) -> bytes:
    import os
    import re
    import shlex
    import subprocess
    import time

    start = time.time()
    subprocess.run(["nvidia-smi"], check=True)

    cpuinfo = Path("/proc/cpuinfo").read_text()
    cpu_model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    print(f"cpu model: {cpu_model.group(1) if cpu_model else 'unknown'}")
    print(f"host cpus: {os.cpu_count()}, usable cpus (affinity): {len(os.sched_getaffinity(0))}")
    print(f"cgroup cpu.max (quota period): {cpu_max.read_text().strip() if cpu_max.exists() else 'n/a'}")
    print(f"load average: {os.getloadavg()}")
    proc = subprocess.run(["uv", "run", "--frozen", *shlex.split(command)], cwd=REMOTE)

    # Ship back every file under results/ created or modified by the command, even if it failed.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in Path(REMOTE, "results").rglob("*"):
            if path.is_file() and path.stat().st_mtime >= start:
                tar.add(path, arcname=str(path.relative_to(REMOTE)))
    if proc.returncode != 0:
        print(f"command exited with code {proc.returncode}")
    return buf.getvalue()


@app.local_entrypoint()
def main(command: str = "pytest -q", gpu: str = ""):
    fn = run_command.with_options(gpu=gpu) if gpu else run_command
    archive = fn.remote(command)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        names = tar.getnames()
        tar.extractall(REPO, filter="data")
    for name in names:
        print(f"synced {name}")
