# Benchmark (2026-11-17, NVIDIA GeForce RTX 3080, ctx 131072)

## Short-context decode (tok/s; 500 generated tokens; prompts: ja / story / code; greedy + temp 0.7 x 3 seeds)

| config | max | mean | min |
|---|---|---|---|
| 速度優先 (`--kv q3 --spec-relax 4 --skip-ffn 29,10`) | 80.5 | 64.2 | 54.2 |
| 厳密 (q4 KV, exact sampling) | 72.8 | 56.1 | 47.4 |

## Prefill and long-context decode (速度優先 config, temp 0.7, 200 generated tokens)

| context | prefill | prefill tok/s | decode tok/s |
|---|---|---|---|
| 2048 | 1.72 s | 1189.4 | 72.63 |
| 8192 | 6.81 s | 1202.9 | 66.17 |
| 32768 | 32.25 s | 1016.1 | 58.52 |
| 65536 | 78.67 s | 833.0 | 40.91 |
| 127271 | 204.75 s | 621.6 | 28.26 |
| 127271 (q4 KV, exact) | 175.03 s | 727.1 | 24.22 |
