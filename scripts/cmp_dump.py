import re, sys
# parse eval-callback dump: tensor name -> list of rows (first3 + last3 values)
def parse_ref(path):
    out = {}; name = None; rows = []
    for line in open(path):
        m = re.match(r'common_debug_cb_eval:\s+(\S+) = ', line)
        if m:
            if name and rows and name not in out: out[name] = rows
            name, rows = m.group(1), []
            continue
        if name and line.strip().startswith('[') and '...' in line:
            vals = [float(x) for x in re.findall(r'-?\d+\.\d+(?:e[-+]?\d+)?', line)]
            rows.append(vals)
    if name and rows and name not in out: out[name] = rows
    return out
ref = parse_ref(sys.argv[1])
mine = {}
for line in open(sys.argv[2]):
    p = line.split()
    if p and p[0] == 'ROW':
        mine.setdefault(p[1], []).append([float(x) for x in p[2:]])
for name in sorted(mine, key=lambda n: (int(n.rsplit('-',1)[1]), n)):
    if name not in ref: continue
    r, m = ref[name], mine[name]
    n = min(len(r), len(m)); err = 0; mag = 0
    for a, b in zip(r[:n], m[:n]):
        for x, y in zip(a, b): err = max(err, abs(x-y)); mag = max(mag, abs(x))
    print(f"{name:22s} rows={n} max_abs_err={err:.4f} max_mag={mag:.3f} rel={err/(mag+1e-9):.4f}")
