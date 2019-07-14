# README.md

This document contains environment description and instructions for installation and training. For the solution see the [Report](./Report.md) document.

## Tennis environment

Two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space is a vector of 24 numbers corresponding to the position and velocity of the ball and racket during last 3 frames.

Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). The score for an episode is calculated as follows:

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores.
* We then take the maximum of these 2 scores. This is a single score for an episode.

## Installation of the Environment

All scripts should be run from the root directory of the project.

* `install.sh` will download all the environements.
* Install Python libraries from the [Udacity RL project](https://github.com/udacity/deep-reinforcement-learning);
* If you would like to load Tensorboard graphs of the trained agent or other checkpoints from my storage bucket, you will also need the [gsutils](https://cloud.google.com/storage/docs/gsutil_install) tool for getting files from Google Cloud Platform.

## Training

Run the `train-tennis.sh` script from the the root folder of the repo.

Trained models are stored in the `checkpoins` folder and TensorBoard statistics is written into the `train` folder.

Training is done through iterations. Number of iterations is defined via the `iterations` parameter. Each iteration consists of 2 phases:

1. Training. It runs at least `steps` steps;
2. Evaluation. Runs at least `eval_steps`. During the evaluation phase network is not trained, replay buffer is not updated and the noisy layers don't change their noise parameters. Only the reward is calculated.

For getting a full list of training parameters run `python train.py`.

For viewing TensorBoard graphs run `tensorboard --logdir=./train` from the root directory of the repo.

## Running trained agent

Video of the original trained agent from the [solution](./Report.md):

![play](https://github.com/vladhc/rl-udacity/raw/master/p3_continuous/tennis.gif "Agent playing Tennis environment")

For visualization of the trained agent use the `play.py` script from the root directory of the git repo:

`python play.py --checkpoint path-to-checkpoint.pth`

This renders the environment, runs multiple episodes and outputs the average reward.

Repository contains an already trained agent. In order to run it:
* Copy the saved network from the repo to a local machine: `wget https://github.com/vladhc/rl-udacity/raw/master/p3_continuous/tennis-ppo-3262.pth`
* Run visualization: `python play.py --checkpoint tennis-ppo-3262.pth`

For a complete description of the solution, please see the [report](./Report.md).
