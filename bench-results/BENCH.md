# Benchmark (short context 2026-11-19, long context 2026-11-18; NVIDIA GeForce RTX 3080, ctx 131072)

## Short-context decode (tok/s; 500 generated tokens; prompts: ja / story / code; greedy + temp 0.7 x 3 seeds)

| config | max | mean | min |
|---|---|---|---|
| 速度優先 (`--kv q3 --spec-relax 2 --skip-ffn 29,10`) | 117.1 | 80.6 | 61.0 |
| 厳密 (q4 KV, exact sampling) | 117.8 | 77.3 | 56.2 |

Raw (ja x4, story x4, code x4; greedy, temp 0.7 seeds 1-3): speed 65.48 64.78 67.18 64.22 / 60.98 69.37 67.43 64.34 / 108.03 108.63 109.34 117.10; exact 64.49 57.46 58.58 60.43 / 57.49 58.49 56.21 59.31 / 117.82 109.25 115.88 112.68.

## Prefill and long-context decode (速度優先 config, temp 0.7, 200 generated tokens)

| context | prefill | prefill tok/s | decode tok/s |
|---|---|---|---|
| 2048 | 1.72 s | 1188.9 | 86.77 |
| 8192 | 6.42 s | 1276.1 | 82.95 |
| 32768 | 29.20 s | 1122.4 | 69.32 |
| 65536 | 68.26 s | 960.1 | 55.66 |
| 127271 | 162.07 s | 785.3 | 44.99 |
| 127271 (q4 KV, exact; before the attention changes) | 169.16 s | 752.4 | 26.03 |
| 254542 (ctx 262144) | 459.14 s | 554.4 | 23.70 |
