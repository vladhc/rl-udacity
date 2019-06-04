#!/bin/bash

set -eu

mkdir -p checkpoints
mkdir -p train

STEPS=500
EVAL_STEPS=3000
ENV="CartPole-v1"
EPS_DECAY=12500
SESSION="pole"

python train.py --agent "ppo" \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --env_count 20 \
  --learning_rate 0.0001 \
  --iterations 100

python train.py --double \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --epsilon_end 0.01 --epsilon_decay $EPS_DECAY \
  --iterations 100 --learning_rate 0.0001

python train.py --agent "actor-critic" \
  --sess "$SESSION" --env "$ENV" --gcp \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --iterations 100 --learning_rate 0.0003

python train.py --agent "actor-critic" \
  --sess "$SESSION-multiple20" --env "$ENV" --env_count 20 \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --epsilon_end 0.01 --epsilon_decay $EPS_DECAY \
  --iterations 100 --learning_rate 0.0003

python train.py --agent "actor-critic" \
  --sess "$SESSION-multiple4" --env "$ENV" --env_count 4 \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --iterations 100 --learning_rate 0.0003

python train.py \
  --sess "$SESSION-every4" --env "$ENV" \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --epsilon_end 0.01 --epsilon_decay $EPS_DECAY \
  --ref_net "pole-ref" --iterations 100 --learning_rate 0.0004 --train_freq 4

python train.py --noisy \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --ref_net "pole-ref" --iterations 100 --learning_rate 0.0001

python train.py --priority \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --epsilon_end 0.01 --epsilon_decay $EPS_DECAY \
  --ref_net "pole-ref" --iterations 100 --learning_rate 0.0001

python train.py --dueling \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --epsilon_end 0.01 --epsilon_decay $EPS_DECAY \
  --ref_net "pole-ref" --iterations 100 --learning_rate 0.0001

python train.py --soft \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --epsilon_end 0.01 --epsilon_decay $EPS_DECAY \
  --ref_net "pole-ref" --iterations 100 --learning_rate 0.0001

python train.py --double --dueling --noisy --priority --soft \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --ref_net "pole-ref" --iterations 100 --learning_rate 0.0001

python train.py --agent "reinforce" \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --iterations 100 --learning_rate 0.001

python train.py --agent "reinforce" --baseline \
  --sess "$SESSION" --env "$ENV" \
  --steps $STEPS --eval_steps $EVAL_STEPS \
  --baseline_learning_rate 0.004 \
  --iterations 100 --learning_rate 0.001
