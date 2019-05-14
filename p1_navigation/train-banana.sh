#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

python train.py --sess banana-no-noisy --env banana --double --dueling --priority --steps 2000
python train.py --sess banana          --env banana --double --dueling --noisy --priority --steps 2000
