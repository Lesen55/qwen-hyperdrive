import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
f = r.fields['tokenizer.ggml.tokens']
toks = [bytes(f.parts[i]).decode('utf-8') for i in f.data]
bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256+n); n += 1
dec = {chr(c): b for b, c in zip(bs, cs)}
ids = [int(x) for x in open(sys.argv[2]).read().split()]
a, b = int(sys.argv[3]), int(sys.argv[4])
out = bytearray()
for i in ids[a:b]: out += bytes(dec[ch] for ch in toks[i])
sys.stdout.buffer.write(bytes(out))
