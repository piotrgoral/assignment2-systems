"""2.1.5 mixed_precision_accumulation: sum 0.01 a thousand times with fp32 / fp16 accumulators and addends.

Example:
    uv run python -m cs336_systems.mixed_precision_accumulation --output results.jsonl

The four loops are the ones from the handout; the exact answer is 10.
"""

import argparse
import json

import torch

N = 1000
EXACT = 10.0


def accumulate(acc_dtype: torch.dtype, add_dtype: torch.dtype, explicit_cast: bool = False) -> torch.Tensor:
    s = torch.tensor(0, dtype=acc_dtype)
    for _ in range(N):
        x = torch.tensor(0.01, dtype=add_dtype)
        s += x.type(acc_dtype) if explicit_cast else x
    return s


CASES = [
    ("fp32 acc += fp32", dict(acc_dtype=torch.float32, add_dtype=torch.float32)),
    ("fp16 acc += fp16", dict(acc_dtype=torch.float16, add_dtype=torch.float16)),
    ("fp32 acc += fp16 (implicit upcast)", dict(acc_dtype=torch.float32, add_dtype=torch.float16)),
    ("fp32 acc += fp16.type(fp32)", dict(acc_dtype=torch.float32, add_dtype=torch.float16, explicit_cast=True)),
]


def main(argv: list[str] | None = None) -> list[dict]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", help="Append one JSON line per case to this file.")
    args = p.parse_args(argv)

    for dtype in (torch.float32, torch.float16):
        print(f"0.01 stored as {dtype}: {torch.tensor(0.01, dtype=dtype).item():.10f}")

    results = []
    for label, kwargs in CASES:
        s = accumulate(**kwargs)
        value = s.item()
        result = dict(case=label, result_dtype=str(s.dtype), value=value, abs_error=abs(value - EXACT))
        results.append(result)
        print(f"{label:<36} {value:.6f}  (|err| = {result['abs_error']:.6f}, dtype {s.dtype})")

    if args.output:
        with open(args.output, "a") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
    return results


if __name__ == "__main__":
    main()
