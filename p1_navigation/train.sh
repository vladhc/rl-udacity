#!/bin/bash

set -eu

ENV="CartPole-v1"
STEPS=400

mkdir -p checkpoins
mkdir -p train

SESSION="noisy"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --noisy --steps $STEPS

SESSION="baseline"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --steps $STEPS

SESSION="double"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --double --steps $STEPS

SESSION="dueling"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --dueling --steps $STEPS

SESSION="double-dueling"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --double --dueling --steps $STEPS
