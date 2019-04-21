#!/bin/bash

set -eu

ENV="cartpole"
SESSION="baseline"

mkdir -p checkpoins
mkdir -p train
rm -rf train/$SESSION

python train.py --sess "$SESSION" --env "$ENV"
