#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

STEPS=100
SESSION="test"

echo "Test: Running one iteration on all environments"

# Banana

python train.py \
  --sess "$SESSION" --env "banana" \
  --steps $STEPS --eval_steps $STEPS \
  --iterations 1

python train.py --agent "reinforce" \
  --sess "$SESSION" --env "banana" \
  --steps $STEPS --eval_steps $STEPS \
  --iterations 1

# CartPole

python train.py \
  --sess "$SESSION" --env "CartPole-v1" \
  --steps $STEPS --eval_steps $STEPS \
  --iterations 1

python train.py --agent "reinforce" \
  --sess "$SESSION" --env "CartPole-v1" \
  --steps $STEPS --eval_steps $STEPS \
  --iterations 1 --learning_rate 0.00002

# LunarLander

python train.py \
  --sess "$SESSION" --env "LunarLander-v2" \
  --steps $STEPS --eval_steps $STEPS \
  --iterations 1

python train.py --agent "reinforce" \
  --sess "$SESSION" --env "LunarLander-v2" \
  --steps $STEPS --eval_steps $STEPS \
  --iterations 1

# Car

python train.py \
  --sess "$SESSION" --env "MountainCar-v0" \
  --steps $STEPS --eval_steps $STEPS \
  --iterations 1

python train.py --agent "reinforce" \
  --sess "$SESSION" --env "MountainCar-v0" \
  --steps $STEPS --eval_steps $STEPS \
  --iterations 1
