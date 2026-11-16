#!/usr/bin/env python3
"""Compare qe_dump (our dequant) with gguf-py's numpy dequant for one tensor of every quant type."""
import subprocess, sys, numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
model, exe = sys.argv[1], sys.argv[2]
r = GGUFReader(model)
seen = {}
for t in r.tensors:
    if t.tensor_type.name not in seen: seen[t.tensor_type.name] = t
ok = True
for name, t in sorted(seen.items()):
    nrows = 4
    ne0 = int(t.shape[0])
    raw = np.asarray(t.data).reshape(-1)
    rowbytes = raw.size // int(np.prod(t.shape[1:])) if len(t.shape) > 1 else raw.size
    ref = dequantize(np.asarray(t.data)[:nrows] if len(t.shape) > 1 else np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(-1)
    out = subprocess.run([exe, model, t.name, str(nrows)], capture_output=True, check=True).stdout
    mine = np.frombuffer(out, dtype=np.float32)
    ref = ref[:mine.size]
    err = np.max(np.abs(mine - ref)) / (np.max(np.abs(ref)) + 1e-30)
    status = "OK" if err < 1e-5 else "MISMATCH"
    if err >= 1e-5: ok = False
    print(f"{name:8s} {t.name:32s} n={mine.size:7d} rel_max_err={err:.2e} {status}")
sys.exit(0 if ok else 1)
