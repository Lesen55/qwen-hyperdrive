# Reference baseline (llama.cpp a946eb5, CUDA, same machine), ctx=131072, KV q4_0, FA on, MTP draft

| config | prompt tok/s | decode tok/s | MTP accept |
|---|---|---|---|
| `--fit on` (auto: whole layers on CPU, FA over K/V disabled) | 3.8 | 2.07 (temp 1) | 32/90 |
| tuned: `-ngl 99`, all 64 FFN on CPU (`-ot`), CUDA graphs disabled (else OOM) | 7.2 (24-tok prompt) | 2.66 (greedy) / 2.46 (temp 1) | 122/229 |

Any FFN layer kept on GPU OOMs at 128K. Decode at ~376 ms/token is ~3x above the DRAM floor
(5.8 GB host FFN / 46 GB/s = 125 ms): the CPU IQ kernels are compute-bound.
