#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for tag in $GUARDS; do
  python -u finetune_multihead.py --model "$(ft_model "$tag")" \
      --dataset wildguardmix toxicchat aegis2 --loss supcon \
      --output_dir "models/multihead-half-supcon-${tag}-all-layer-dedup" \
      2>&1 | tee "logs/train_heads_${tag}.log"
done
