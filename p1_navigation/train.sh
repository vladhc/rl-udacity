#!/bin/bash

set -eu

ENV="cartpole"
SESSION="double-rewardmod"

mkdir -p checkpoins
mkdir -p train

rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV"

SESSION="double-dueling-rewardmod"
rm -rf train/$SESSION
python train.py --sess "$SESSION" --env "$ENV" --dueling
