# 2.1.5 mixed_precision_accumulation — commands

Script: `cs336_systems/mixed_precision_accumulation.py` (the four handout loops, CPU only, runs in under a second).

```bash
rm -f results/data/2_1_5_mixed_precision_accumulation.jsonl
uv run python -m cs336_systems.mixed_precision_accumulation \
  --output results/data/2_1_5_mixed_precision_accumulation.jsonl \
  2>&1 | tee results/data/2_1_5_mixed_precision_accumulation.log
```
