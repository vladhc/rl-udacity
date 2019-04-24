#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train
STEPS=400

ENV="LunarLander-v2"

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


ENV="CartPole-v0"

SESSION="pole-baseline"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --steps $STEPS

SESSION="pole-double"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --double --steps $STEPS

SESSION="pole-noisy"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --noisy --steps $STEPS

SESSION="pole-dueling"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --dueling --steps $STEPS

SESSION="pole-double-noisy"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --double --noisy --steps $STEPS

SESSION="pole-dueling-noisy"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --dueling --noisy --steps $STEPS

SESSION="pole-double-dueling"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --double --dueling --steps $STEPS

SESSION="pole-double-dueling-noisy"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --double --dueling --noisy --steps $STEPS
