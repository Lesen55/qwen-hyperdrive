#!/usr/bin/env bash
# One-shot Vulkan report for a target GPU (e.g. Intel Arc 130V, AMD RX 7900 XTX).
#   nix run .#vulkan-report -- <model.gguf>      or      scripts/vk_device_report.sh <model.gguf> [bin-dir]
# Re-runs the start-up autotuner (QE_VK_RETUNE=1), the shader unit tests, decode / prefill timings and
# a short perplexity, and writes everything to vk-report-<device>.txt for comparison / further tuning.
set -u
model="${1:?usage: vk_device_report.sh <model.gguf> [dir with qwen35-engine-vk, qe_vk_test, qe_tokenize]}"
bin="${2:-$(dirname "$(readlink -f "$0")")}"
for b in qwen35-engine-vk qe_vk_test; do
  [ -x "$bin/$b" ] || { echo "missing $bin/$b (build with: nix build .#vulkan)" >&2; exit 1; }
done
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
dev=$(vulkaninfo --summary 2>/dev/null | grep -m1 deviceName | sed 's/.*= //; s/[^A-Za-z0-9]/_/g')
out="vk-report-${dev:-unknown}.txt"
{
  echo "== $(date -Is)  model: $(basename "$model")"
  echo "== vulkaninfo"; vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|deviceType|driverName|driverInfo|apiVersion' | head -8
  echo "== memory"; free -g | head -2
  echo "== shader unit tests"; "$bin/qe_vk_test" "$model" 2>&1 | tail -20
  # synthetic token ids (the vocabulary of every qwen35 model is > 150k): decode and prefill timings
  awk 'BEGIN{srand(1); for(i=0;i<32;i++) printf "%d ", 1000+int(rand()*100000)}' > "$tmp/p32.ids"
  awk 'BEGIN{srand(2); for(i=0;i<2048;i++) printf "%d ", 1000+int(rand()*100000)}' > "$tmp/p2k.ids"
  awk 'BEGIN{srand(3); for(i=0;i<8192;i++) printf "%d ", 1000+int(rand()*100000)}' > "$tmp/p8k.ids"
  echo "== autotune (forced re-measure) + decode 256 tokens, 128K context setting"
  QE_VK_RETUNE=1 "$bin/qwen35-engine-vk" gen -m "$model" --ids "$tmp/p32.ids" -n 256 -c 131072 --temp 0 -v 2>&1 |
    grep -E '^\[hw\]|^\[plan\]|^\[tune\]|^prompt'
  for ids in p2k p8k; do
    echo "== prefill $ids"
    "$bin/qwen35-engine-vk" gen -m "$model" --ids "$tmp/$ids.ids" -n 64 -c 32768 --temp 0 2>&1 | grep -E '^prompt'
  done
  echo "== GPU time per shader (prefill 2K + 64 decode tokens)"
  QE_VK_PROF=1 "$bin/qwen35-engine-vk" gen -m "$model" --ids "$tmp/p2k.ids" -n 64 -c 32768 --temp 0 2>&1 | grep -A16 vkprof
  echo "== tune cache"; cat ~/.cache/qwen-hyperdrive/vk-tune-*.txt 2>/dev/null
} 2>&1 | tee "$out"
echo "report written to $out" >&2
