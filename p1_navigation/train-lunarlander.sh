#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

STEPS=2000
ENV="LunarLander-v2"

SESSION="lunar-double-dueling-noisy"
python train.py --sess "$SESSION" --env "$ENV" --double --dueling --noisy --steps $STEPS

SESSION="lunar-double-dueling-noisy-priority"
python train.py --sess "$SESSION" --env "$ENV" --double --dueling --noisy --priority --steps $STEPS

SESSION="lunar-priority"
python train.py --sess "$SESSION" --env "$ENV" --priority --steps $STEPS

SESSION="lunar-double-priority"
python train.py --sess "$SESSION" --env "$ENV" --double --priority --steps $STEPS

SESSION="lunar-baseline"
python train.py --sess "$SESSION" --env "$ENV" --steps $STEPS

SESSION="lunar-double"
python train.py --sess "$SESSION" --env "$ENV" --double --steps $STEPS

SESSION="lunar-noisy"
python train.py --sess "$SESSION" --env "$ENV" --noisy --steps $STEPS

SESSION="lunar-dueling"
python train.py --sess "$SESSION" --env "$ENV" --dueling --steps $STEPS

SESSION="lunar-double-noisy"
python train.py --sess "$SESSION" --env "$ENV" --double --noisy --steps $STEPS

SESSION="lunar-dueling-noisy"
python train.py --sess "$SESSION" --env "$ENV" --dueling --noisy --steps $STEPS

SESSION="lunar-double-dueling"
python train.py --sess "$SESSION" --env "$ENV" --double --dueling --steps $STEPS

SESSION="lunar-double-dueling-noisy"
python train.py --sess "$SESSION" --env "$ENV" --double --dueling --noisy --steps $STEPS
