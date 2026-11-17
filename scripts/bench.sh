#!/usr/bin/env bash
# Benchmark: short-context decode (max / mean / min over prompts x sampling settings) for two
# configurations, prefill rates, and long-context prefill + decode.  Writes bench-results/BENCH.md.
set -u
cd "$(dirname "$0")/.."
BIN=${BIN:-./build/qwen35-engine}
MODEL=${MODEL:-models.gguf}
export LD_LIBRARY_PATH=/run/opengl-driver/lib
OUT=bench-results/BENCH.md
FAST="--kv q3 --spec-relax 4 --skip-ffn 29,10"
EXACT=""
run() { $BIN gen -m "$MODEL" -c 131072 "$@" 2>&1; }
dec() { grep -o 'gen [0-9]* tok in [0-9.]* s ([0-9.]* tok/s)' | grep -o '([0-9.]*' | tr -d '('; }
pf() { grep -o 'prompt [0-9]* tok in [0-9.]* s ([0-9.]* tok/s)' | sed -E 's/prompt ([0-9]+) tok in ([0-9.]+) s \(([0-9.]+) tok\/s\)/\1 \2 \3/'; }
stats() { awk '{a[NR]=$1; s+=$1} END {mx=a[1]; mn=a[1]; for(i in a){if(a[i]>mx)mx=a[i]; if(a[i]<mn)mn=a[i]} printf "%.1f | %.1f | %.1f", mx, s/NR, mn}'; }
{
echo "# Benchmark ($(date +%F), $(nvidia-smi --query-gpu=name --format=csv,noheader), ctx 131072)"
echo
echo "## Short-context decode (tok/s; 500 generated tokens; prompts: ja / story / code; greedy + temp 0.7 x 3 seeds)"
echo
echo "| config | max | mean | min |"
echo "|---|---|---|---|"
} > $OUT
for cfg in FAST EXACT; do
  args=${!cfg}
  vals=()
  for p in ja story code; do
    vals+=("$(run --ids bench-results/data/$p.ids -n 500 --temp 0 $args | dec)")
    for sd in 1 2 3; do vals+=("$(run --ids bench-results/data/$p.ids -n 500 --temp 0.7 --seed $sd $args | dec)"); done
  done
  echo "$cfg: ${vals[*]}" >&2
  name=$([ $cfg = FAST ] && echo "速度優先 (\`$FAST\`)" || echo "厳密 (q4 KV, exact sampling)")
  echo "| $name | $(printf '%s\n' "${vals[@]}" | stats) |" >> $OUT
done
{
echo
echo "## Prefill and long-context decode (速度優先 config, temp 0.7, 200 generated tokens)"
echo
echo "| context | prefill | prefill tok/s | decode tok/s |"
echo "|---|---|---|---|"
} >> $OUT
for n in 2048 8192 32768 65536 127271; do
  awk -v n=$n '{for(i=1;i<=n && i<=NF;i++) printf "%s ", $i; print ""}' bench-results/data/passkey.ids > /tmp/bench_$n.ids
  o=$(run --ids /tmp/bench_$n.ids -n 200 --temp 0.7 --seed 1 $FAST)
  read -r pn ps pt <<< "$(echo "$o" | pf)"
  d=$(echo "$o" | dec)
  echo "ctx $n: $pn $ps $pt $d" >&2
  echo "| $pn | ${ps} s | ${pt} | ${d} |" >> $OUT
done
o=$(run --ids bench-results/data/passkey.ids -n 200 --temp 0.7 --seed 1)
read -r pn ps pt <<< "$(echo "$o" | pf)"
echo "| $pn (q4 KV, exact) | ${ps} s | ${pt} | $(echo "$o" | dec) |" >> $OUT
cat $OUT
