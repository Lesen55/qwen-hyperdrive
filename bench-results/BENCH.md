# Benchmark (2026-11-18, NVIDIA GeForce RTX 3080, ctx 131072)

## Short-context decode (tok/s; 500 generated tokens; prompts: ja / story / code; greedy + temp 0.7 x 3 seeds)

| config | max | mean | min |
|---|---|---|---|
| 速度優先 (`--kv q3 --spec-relax 2 --skip-ffn 29,10`) | 103.4 | 71.9 | 57.8 |
| 厳密 (q4 KV, exact sampling) | 104.9 | 69.7 | 49.7 |

## Prefill and long-context decode (速度優先 config, temp 0.7, 200 generated tokens)

| context | prefill | prefill tok/s | decode tok/s |
|---|---|---|---|
| 2048 | 1.68 s | 1216.6 | 86.47 |
| 8192 | 6.60 s | 1241.0 | 83.17 |
| 32768 | 31.17 s | 1051.3 | 62.45 |
| 65536 | 76.14 s | 860.8 | 51.62 |
| 127271 | 198.62 s | 640.8 | 34.18 |
| 127271 (q4 KV, exact) | 169.16 s | 752.4 | 26.03 |
