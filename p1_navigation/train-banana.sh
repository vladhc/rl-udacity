#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

python train.py \
  --double --dueling --noisy --priority \
  --sess banana --env banana \
  --steps 200 --eval_steps 0 \
  --iterations 4000
