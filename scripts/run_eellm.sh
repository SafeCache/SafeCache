#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for tag in $GUARDS; do
  model=$(ft_model "$tag")
  exits="models/eellm-exits-$(basename "$model")-trainfull"
  if [ ! -f "$exits/config.json" ]; then
    echo "=== [$tag] EE-Tuning exit layers -> $exits ==="
    python -u eellm_tune.py --model "$model" --cache-dataset $CACHE_SETS --train-full \
        --out "$exits" 2>&1 | tee "logs/eellm_tune_${tag}.log"
  fi
  echo "=== [$tag] EE-LLM confidence-threshold sweep ==="
  python -u eellm_eval.py --model "$model" --heads "$exits" \
      --test-dataset $TEST_SETS --sim-step 0.01 --sim-min 0.0 \
      --summary-file "results/result-eellm/results_sweep_${tag}_eellm_fine" \
      2>&1 | tee "logs/eellm_sweep_${tag}.log"
done
