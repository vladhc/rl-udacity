#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

STEPS=300
ENV="CartPole-v0"

SESSION="pole-baseline"
python train.py --sess "$SESSION" --env "$ENV" --steps $STEPS

SESSION="pole-double"
python train.py --sess "$SESSION" --env "$ENV" --double --steps $STEPS

SESSION="pole-priority"
python train.py --sess "$SESSION" --env "$ENV" --priority --steps $STEPS

SESSION="pole-noisy"
python train.py --sess "$SESSION" --env "$ENV" --noisy --steps $STEPS

SESSION="pole-dueling"
python train.py --sess "$SESSION" --env "$ENV" --dueling --steps $STEPS

SESSION="pole-double-noisy"
python train.py --sess "$SESSION" --env "$ENV" --double --noisy --steps $STEPS

SESSION="pole-dueling-noisy"
python train.py --sess "$SESSION" --env "$ENV" --dueling --noisy --steps $STEPS

SESSION="pole-double-dueling"
python train.py --sess "$SESSION" --env "$ENV" --double --dueling --steps $STEPS

SESSION="pole-double-dueling-noisy"
python train.py --sess "$SESSION" --env "$ENV" --double --dueling --noisy --steps $STEPS

SESSION="pole-double-dueling-noisy-priority"
python train.py --sess "$SESSION" --env "$ENV" --double --dueling --noisy --priority --steps $STEPS
