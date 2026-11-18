# Benchmark (short context 2026-11-19, long context 2026-11-18; NVIDIA GeForce RTX 3080, ctx 131072)

## Short-context decode (tok/s; 500 generated tokens; prompts: ja / story / code; greedy + temp 0.7 x 3 seeds)

| config | max | mean | min |
|---|---|---|---|
| 速度優先 (`--kv q3 --spec-relax 2 --skip-ffn 29,10`) | 107.2 | 76.0 | 57.0 |
| 厳密 (q4 KV, exact sampling) | 105.2 | 70.0 | 51.9 |

Raw (ja x4, story x4, code x4; greedy, temp 0.7 seeds 1-3): speed 57.03 60.43 62.02 66.17 / 58.54 60.15 63.48 69.03 / 103.17 105.21 99.14 107.16; exact 59.34 57.46 55.01 52.18 / 55.55 58.50 51.93 53.70 / 99.24 97.21 105.20 95.22.

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
