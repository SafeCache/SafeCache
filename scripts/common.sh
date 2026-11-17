#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FT_SUFFIX="wildguardmix+toxicchat+aegis2-ep1-lr0.0002-ft"
declare -A BASE=(
  [lg3]="Llama-Guard-3-8B"
  [lg4]="Llama-Guard-4-12B"
  [sg]="ShieldGemma-9B"
  [ag]="Aegis-LlamaGuard-Permissive-1.0"
)
GUARDS=${GUARDS:-"lg3 lg4 sg ag"}
TEST_SETS="wildguardmix toxicchat aegis2 simple openai harmbench"
CACHE_SETS="wildguardmix aegis2 toxicchat"
ft_model() { echo "./models/${BASE[$1]}-${FT_SUFFIX}"; }
mkdir -p logs results/result-supcon-frac30 results/result-eellm results/result-gptcache results/prep
