#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

python train.py \
  --sess "pong-dqn" \
  --env "PongNoFrameskip-v4" \
  --steps 1000000 \
  --epsilon_decay 30000 \
  --epsilon_start 1.0 \
  --epsilon_end 0.01 \
  --gamma 0.99 \
  --learning_rate 0.0001 \
  --target_update_freq 1000 \
  --replay_buffer_size 100000 \
  --batch_size 32 \
  --hidden_units 512

python train.py --sess banana-no-noisy --env banana --double --dueling --priority --steps 2000
python train.py --sess banana-2 --env banana --double --dueling --noisy --priority --steps 2000


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


ENV="CartPole-v0"

SESSION="pole-baseline"
python train.py --sess "$SESSION" --env "$ENV" --steps $STEPS

SESSION="pole-double"
python train.py --sess "$SESSION" --env "$ENV" --double --steps $STEPS

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
