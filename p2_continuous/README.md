# README.md

This document contains environment description and instructions for installation and training. For the solution see the [Report](./Report.md) document.

## Reacher environment

A double-jointed arm should move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved, when the average score over 100 episodes is at least +30.

There is also a distributed version of the environment, which 20 has simultaneously running agents.

## Installation of the Environment

All scripts should be run from the root directory of the project.

* `install.sh` will download all the environements.
* Install Python libraries from the [Udacity RL project](https://github.com/udacity/deep-reinforcement-learning);
* If you would like to load Tensorboard graphs of the trained agent or other checkpoints from my storage bucket, you will also need the [gsutils](https://cloud.google.com/storage/docs/gsutil_install) tool for getting files from Google Cloud Platform.

## Training

Run the `train-reacher.sh` script from the the root folder of the repo.

Trained models are stored in the `checkpoins` folder and TensorBoard statistics is written into the `train` folder.

Training is done through iterations. Number of iterations is defined via the `iterations` parameter. Each iteration consists of 2 phases:

1. Training. It runs at least `steps` steps;
2. Evaluation. Runs at least `eval_steps`. During the evaluation phase network is not trained, replay buffer is not updated and the noisy layers don't change their noise parameters. Only the reward is calculated.

For getting a full list of training parameters run `python train.py`.

For viewing TensorBoard graphs run `tensorboard --logdir=./train` from the root directory of the repo.

## Running trained agent

Video of the original trained agent from the [solution](./Report.md):

![play](https://github.com/vladhc/rl-udacity/raw/master/p2_continuous/reacher.gif "Agent playing Reacher environment")

For visualization of the trained agent use the `play.py` script from the root directory of the git repo:

`python play.py --checkpoint path-to-checkpoint.pth`

This renders the environment, runs multiple episodes and outputs the average reward.

Repository contains an already trained agent. In order to run it:
* Copy the saved network from the repo to a local machine: `wget https://github.com/vladhc/rl-udacity/raw/master/p2_continuous/reacher-ppo-643.pth`
* Run visualization: `python play.py --checkpoint reacher-ppo-643.pth`

For a complete description of the solution, please see the [report](./Report.md).
