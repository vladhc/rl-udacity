#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

STEPS=500
EVAL_STEPS=3000
ENV="CartPole-v1"
SESSION="pole"

python train.py \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS \
  --eval_steps $EVAL_STEPS

python train.py --noisy \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS \
  --eval_steps $EVAL_STEPS

python train.py --double \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS \
  --eval_steps $EVAL_STEPS

python train.py --priority \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS \
  --eval_steps $EVAL_STEPS

python train.py --dueling \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS \
  --eval_steps $EVAL_STEPS

python train.py --double --noisy \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS \
  --eval_steps $EVAL_STEPS

python train.py --dueling --noisy \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS \
  --eval_steps $EVAL_STEPS

python train.py --double --dueling \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS \
  --eval_steps $EVAL_STEPS

python train.py --double --dueling --noisy \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS \
  --eval_steps $EVAL_STEPS

python train.py --double --dueling --noisy --priority \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS \
  --eval_steps $EVAL_STEPS
