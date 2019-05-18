#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

STEPS=1200
ITERS=400
EVAL_STEPS=6000
ENV="LunarLander-v2"
SESSION="lunar"


python train.py --soft \
  --sess "$SESSION-every4" --env "$ENV" --iterations $ITERS --gcp \
  --train_freq 4 --learning_rate 0.0004 \
  --steps $STEPS --eval_steps $EVAL_STEPS

python train.py --soft \
  --sess "$SESSION" --env "$ENV" --iterations $ITERS --gcp \
  --steps $STEPS --eval_steps $EVAL_STEPS

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
