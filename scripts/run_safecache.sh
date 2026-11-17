#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ALPHAS="0.002 0.004 0.006 0.008 0.01 0.012 0.014 0.016 0.018 0.02"
COMMON=(--cache-dataset $CACHE_SETS
        --layerwise --select-layers --select-mode topk --select-frac 0.3
        --select-early-weight 0.1 --index-type ivfpq
        --cache-emb --cache-guard-labels)
for tag in $GUARDS; do
  model=$(ft_model "$tag")
  heads="models/multihead-half-supcon-${tag}-all-layer-dedup/epoch_1"
  echo "=== [$tag] SafeCache: building cache-split embeddings and guard labels ==="
  python -u semantic_cache_eval.py --model "$model" --heads "$heads" "${COMMON[@]}" \
      --test-dataset simple --skip-lg-only --fpr-alpha 0.02 \
      --summary-file "results/prep/prep_${tag}" \
      2>&1 | tee "logs/safecache_prep_${tag}.log"
  echo "=== [$tag] SafeCache: static-cache sweep ==="
  python -u semantic_cache_eval.py --model "$model" --heads "$heads" "${COMMON[@]}" \
      --test-dataset $TEST_SETS --fpr-alpha $ALPHAS --per-example-log \
      --summary-file "results/result-supcon-frac30/results_sweep_${tag}-supcon-frac30" \
      2>&1 | tee "logs/safecache_sweep_${tag}.log"
done
