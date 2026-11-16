#!/usr/bin/env bash
# Tuned reference (llama.cpp) server: all layers on GPU except FFN of layers [FIRST..63] on CPU.
FIRST=${1:-14}
CTX=${2:-131072}
regex="blk\.($(seq -s'|' $FIRST 63))\.ffn_(up|gate|down)\.weight=CPU"
exec ./result-reference/bin/llama-server -m models.gguf -c $CTX -fa on -ctk q4_0 -ctv q4_0 -ngl 99 -ot "$regex" \
  -t 6 --spec-type draft-mtp --port 18080 --no-webui -np 1 --fit off
