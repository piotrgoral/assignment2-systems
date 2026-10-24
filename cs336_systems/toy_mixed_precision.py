"""2.1.5 benchmarking_mixed_precision (a)/(b): dtypes inside the handout's ToyModel under autocast.

Example:
    uv run python -m cs336_systems.toy_mixed_precision --dtype fp16 --output results.jsonl

Params start in fp32. The forward pass and the loss run under torch.autocast; backward runs outside it.
Layer outputs are captured with forward hooks.
"""

import argparse
import json

import torch
import torch.nn as nn

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def probe(dtype: torch.dtype, device: torch.device, batch_size: int = 8, in_features: int = 16, out_features: int = 5) -> dict:
    model = ToyModel(in_features, out_features).to(device)
    x = torch.randn(batch_size, in_features, device=device)
    y = torch.randint(0, out_features, (batch_size,), device=device)

    outputs: dict[str, torch.dtype] = {}
    for name in ("fc1", "relu", "ln", "fc2"):
        getattr(model, name).register_forward_hook(lambda _m, _in, out, name=name: outputs.__setitem__(name, out.dtype))

    with torch.autocast(device_type=device.type, dtype=dtype):
        param_dtypes = {n: str(p.dtype) for n, p in model.named_parameters()}
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
    loss.backward()

    return dict(
        autocast_dtype=str(dtype),
        params=param_dtypes,
        fc1_out=str(outputs["fc1"]),
        relu_out=str(outputs["relu"]),
        ln_out=str(outputs["ln"]),
        logits=str(logits.dtype),
        loss=str(loss.dtype),
        grads={n: str(p.grad.dtype) for n, p in model.named_parameters()},
    )


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtype", choices=DTYPES, default="fp16", help="autocast dtype")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", help="Append the result as one JSON line to this file.")
    args = p.parse_args(argv)

    torch.manual_seed(0)
    result = probe(DTYPES[args.dtype], torch.device(args.device))
    print(f"autocast dtype: {result['autocast_dtype']}")
    for key, value in result.items():
        if key != "autocast_dtype":
            print(f"  {key:<8} {value}")

    if args.output:
        with open(args.output, "a") as f:
            f.write(json.dumps(result) + "\n")
    return result


if __name__ == "__main__":
    main()
