#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

python train.py --agent "ppo" \
  --sess reacher --env reacher \
  --steps 1000 --eval_steps 1000 \
  --env_count 1 \
  --ppo_epochs 12 --learning_rate 0.0002 \
  --iterations 8000
