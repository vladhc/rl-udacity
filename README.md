## One more RL library

This set of python files represents a small RL library. It's main point is simplicity - the amount of code should be minimal. It's purpose is to implement algorithms for solving the Udacity RL Nanodegree projects.

This library allows storing results in the Google Cloud Platform. This is beneficial, when we want to use multiple VMs for training, or want to run training for a long period of time.

## Installation
`install.sh` Will download all the environements. Python libraries from the [Udacity RL project](https://github.com/udacity/deep-reinforcement-learning) should be installed beforehand. If you want to get other trained checkpoints from my storage bucket, you will also need the [gsutils](https://cloud.google.com/storage/docs/gsutil_install) tool for getting files from Google Cloud Platofrm.

## Training
Training is done through iterations. Number of iterations is defined via the `iterations` parameter.
Each iteration consists of 2 phases:

1. Training. It runs at least `steps` steps;
2. Evaluation. Runs at least `eval_steps`. During the evaluation phase network is not trained, replay buffer is not updated and the noisy layers don't change their noise parameters. Only the reward is calculated.

For getting a full list of training parameters run `python train.py`.

Trained models are stored in the `checkpoins` folder and TensorBoard statistics is written into the `train` folder.

## Solutions

1. [Bananas](./p1_navigation/README.md)
2. [Reacher](./p2_continuous/README.md)
