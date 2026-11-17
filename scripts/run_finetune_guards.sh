#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
declare -A SCRIPT=([lg3]=finetune_lg3.py [lg4]=finetune_lg4.py [sg]=finetune_sg.py [ag]=finetune_aegis.py)
for tag in $GUARDS; do
  python -u "${SCRIPT[$tag]}" --model "./models/${BASE[$tag]}" \
      --dataset wildguardmix toxicchat aegis2 --epochs 1 --lr 2e-4 \
      2>&1 | tee "logs/finetune_${tag}.log"
done
