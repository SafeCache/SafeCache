#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for tag in $GUARDS; do
  echo "=== [$tag] GPTCache similarity-threshold sweep ==="
  python -u gptcache_baseline_eval.py --model "$(ft_model "$tag")" \
      --cache-dataset $CACHE_SETS --per-example-log \
      --summary-file "results/result-gptcache/results_gptcache_${tag}" \
      2>&1 | tee "logs/gptcache_${tag}.log"
done
