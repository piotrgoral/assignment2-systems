# 2.1.5 mixed_precision_accumulation

**Answer.** Only the pure fp32 loop is close to the exact 10 (10.000134). The pure fp16 loop is far off (9.953125): once the running sum passes 8, the gap between neighbouring fp16 values is 2⁻⁷ ≈ 0.0078, so each added 0.01 gets rounded down to 0.0078 and the sum falls behind. Both mixed loops, with an fp32 accumulator and fp16 addends (implicit or explicit upcast give the same 10.002136), avoid that rounding error, and what's left is just fp16 storing 0.01 as 0.0100021 (1000 × that = 10.0021). So accumulators should be kept in fp32 even when the inputs are fp16.

| accumulator | addend | result | \|error\| |
|---|---|--:|--:|
| fp32 | fp32 | 10.000134 | 0.000134 |
| fp16 | fp16 | 9.953125 | 0.046875 |
| fp32 | fp16 (implicit upcast in `+=`) | 10.002136 | 0.002136 |
| fp32 | fp16 `.type(torch.float32)` | 10.002136 | 0.002136 |

0.01 as stored: fp32 = 0.0099999998, fp16 = 0.0100021362.

Raw data: `results/data/2_1_5_mixed_precision_accumulation.jsonl` (log: `results/data/2_1_5_mixed_precision_accumulation.log`).
