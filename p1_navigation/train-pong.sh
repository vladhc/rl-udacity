#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

python train.py \
  --sess "pong" \
  --env "PongNoFrameskip-v4" \
  --gcp \
  --steps 1000000 \
  --epsilon_decay 30000 \
  --epsilon_start 1.0 \
  --epsilon_end 0.01 \
  --gamma 0.99 \
  --learning_rate 0.0001 \
  --target_update_freq 1000 \
  --replay_buffer_size 100000 \
  --batch_size 32 \
  --hidden_units 512
