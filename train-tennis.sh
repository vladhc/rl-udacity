#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p trajectories
mkdir -p train

python train.py --agent "ppo" \
  --sess tennis --env tennis \
  --steps 1000 --eval_steps 200 \
  --env_count 30 \
  --horizon 500 \
  --ppo_epochs 12 --learning_rate 0.0003 \
  --iterations 8000
