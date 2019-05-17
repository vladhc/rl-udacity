#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

STEPS=1200
ITERS=100
EVAL_STEPS=6000
ENV="LunarLander-v2"
SESSION="lunar"

python train.py \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS --gcp \
  --steps $STEPS --eval_steps $EVAL_STEPS

python train.py --priority \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS --gcp \
  --steps $STEPS --eval_steps $EVAL_STEPS

python train.py --double \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS --gcp \
  --steps $STEPS --eval_steps $EVAL_STEPS

python train.py --noisy \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS --gcp \
  --steps $STEPS --eval_steps $EVAL_STEPS

python train.py --dueling \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS --gcp \
  --steps $STEPS --eval_steps $EVAL_STEPS

python train.py --double --noisy \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS --gcp \
  --steps $STEPS --eval_steps $EVAL_STEPS

python train.py --dueling --noisy \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS --gcp \
  --steps $STEPS --eval_steps $EVAL_STEPS

python train.py --double --dueling \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS --gcp \
  --steps $STEPS --eval_steps $EVAL_STEPS

python train.py --double --priority \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS --gcp \
  --steps $STEPS --eval_steps $EVAL_STEPS

python train.py --double --dueling --noisy --gcp \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS \
  --steps $STEPS --eval_steps $EVAL_STEPS

python train.py --double --dueling --noisy --priority --gcp \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS \
  --steps $STEPS --eval_steps $EVAL_STEPS
