#!/bin/bash

set -eu

ENV="CartPole-v1"

mkdir -p checkpoins
mkdir -p train

SESSION="baseline"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV"

SESSION="double"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --double

SESSION="double-dueling"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --double --dueling
