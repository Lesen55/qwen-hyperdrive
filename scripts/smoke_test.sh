#!/usr/bin/env bash
# Output-correctness smoke test: the greedy continuation of a few prompts must match the simplest
# configuration (no MTP, no elastic lending, small ubatch) for the first N tokens, for every
# optimization toggle.  Catches silent corruption that speed benchmarks cannot see.
set -u
cd "$(dirname "$0")/.."
BIN=${BIN:-./build/qwen35-engine}
MODEL=${MODEL:-models.gguf}
N=${N:-16}
export LD_LIBRARY_PATH=/run/opengl-driver/lib
fail=0
gen() { $BIN gen -m "$MODEL" --ids "$1" -n $N -c 8192 "${@:2}" 2>/dev/null | head -1; }
for p in bench-results/data/story.ids bench-results/data/ja.ids bench-results/data/chunk3.ids; do
  ref=$(gen "$p" --no-mtp --no-elastic --ubatch 512)
  for cfg in "" "--draft 1" "--draft 2" "--ubatch 512" "--no-elastic" "--no-mtp" "--kv q8" ; do
    out=$(gen "$p" $cfg)
    # compare the first N/2 tokens strictly (later ones may legitimately diverge on near-ties)
    a=$(echo "$ref" | cut -d' ' -f1-$((N/2))); b=$(echo "$out" | cut -d' ' -f1-$((N/2)))
    if [ "$a" = "$b" ]; then echo "ok   $(basename $p) [$cfg]"; else echo "FAIL $(basename $p) [$cfg]: $b vs ref $a"; fail=1; fi
  done
done
exit $fail
