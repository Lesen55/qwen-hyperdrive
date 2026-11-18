# Benchmark (2026-11-18, NVIDIA GeForce RTX 3080, ctx 131072)

## Short-context decode (tok/s; 500 generated tokens; prompts: ja / story / code; greedy + temp 0.7 x 3 seeds)

| config | max | mean | min |
|---|---|---|---|
| 速度優先 (`--kv q3 --spec-relax 2 --skip-ffn 29,10`) | 103.4 | 71.9 | 57.8 |
| 厳密 (q4 KV, exact sampling) | 104.9 | 69.7 | 49.7 |

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
