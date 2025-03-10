#!/bin/bash

# Don't try to activate the environment in the script
# The user should activate it before running this script with:
# conda activate cs224n_dfp

# Run DPO training with the best pre-trained model
python sonnet_dpo.py \
  --use_gpu \
  --pretrained_model best_10-1e-05-sonnet.pt \
  --sonnet_pairs_path data/sonnet_pairs_for_dpo.txt \
  --epochs 5 \
  --batch_size 4 \
  --lr 5e-6 \
  --beta 0.1 \
  --temperature 1.0 \
  --top_p 0.9 