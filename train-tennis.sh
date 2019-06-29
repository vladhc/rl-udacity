#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p trajectories
mkdir -p train

python train.py --agent "multippo" \
  --sess tennis2 --env tennis \
  --steps 1000 --eval_steps 200 \
  --env_count 2 \
  --horizon 500 \
  --save_traj \
  --ppo_epochs 10 --learning_rate 0.0003 \
  --iterations 8000
