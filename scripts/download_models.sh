#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p models
get() {
  local repo=$1 dir=$2; shift 2
  if [ -f "models/$dir/config.json" ] || [ -f "models/$dir/adapter_config.json" ]; then
    echo "models/$dir exists, skipping"; return
  fi
  hf download "$repo" --local-dir "models/$dir" "$@"
}
get meta-llama/Llama-Guard-3-8B Llama-Guard-3-8B --exclude "original/*"
get meta-llama/Llama-Guard-4-12B Llama-Guard-4-12B
get google/shieldgemma-9b ShieldGemma-9B
get meta-llama/LlamaGuard-7b LlamaGuard-7b --exclude "*.pth"
get nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0 Aegis-LlamaGuard-Permissive-1.0
