## One more RL library

This set of python files represents a small RL library. It's main point is simplicity - the amount of code should be minimal. It's purpose is to implement algorithms for solving the Udacity RL Nanodegree projects.

Currently in contains implementation of the Deep Q-Learning network with the following improvements:
* [Double Q-Learning](https://arxiv.org/pdf/1509.06461.pdf)
* [Prioritirized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
* [Dueling Network architecture](https://arxiv.org/pdf/1511.06581.pdf)
* [Noisy Networks](https://arxiv.org/pdf/1706.10295.pdf)
* Soft updates

For each improvement a separate benchmark was done on the CartPole and LunarLander environments in order to make sure, that they really add value to the baseline model.

This library allows storing results in the Google Cloud Platform. This is beneficial, when we want to use multiple VMs for training, or want to run training for a long period of time.

## Some beautiful graphs

Can be seen using TensorBoard:
* Training graphs for all benchmarks: `tensorboard --logdir=gs://rl-1/training`
* Training graph for the successful training of the solved Banana environment: `tensorboard --logdir=gs://rl-1/training`

A couple of training experiments on Banana environment. The blue one is the winner:

![training graph](https://github.com/vladhc/rl-udacity/raw/master/p1_navigation/imgs/banana-reward.png "Banana training graph")

A few lunar lander agents. Gray - only "double" feature is enabled, red - only "dueling", orange - "double, dueling, noisy, soft-updates":

![training graph](https://github.com/vladhc/rl-udacity/raw/master/p1_navigation/imgs/lunar-lander.png "Lunar lander training graph")

## Installation
`install.sh` Will download the Banana environement. Python libraries from the [Udacity RL project](https://github.com/udacity/deep-reinforcement-learning) should be installed beforehand. If you want to get other trained checkpoints from my storage bucket, you will also need the [gsutils](https://cloud.google.com/storage/docs/gsutil_install) tool for getting files from Google Cloud Platofrm.

## Training
`train-banana.sh`
This script contains predifined set of hyper parameters. Running this script doesn't mean you will get working agent on the first run. But you will get it eventually, when the random seed would be nice (see also "Other thoughts section").
There are also scripts for training benchmarks: `train-cartpole.sh` and `train-lunarlander.sh`. They train agents turning only certain features. It's convenient for comparing how each feature influences the training.

Training is done through iterations. Number of iterations is defined via the `iterations` parameter.
Each iteration consists of 2 phases:
1. Training. It runs at least `steps` steps;
2. Evaluation. Runs at least `eval_steps`. During the evaluation phase network is not trained, replay buffer is not updated and the noisy layers don't change their noise parameters. Only the reward is calculated.

For getting a full list of training parameters run `python train.py`.

Adam Optimizer was used. I haven't experimented with other optimizers. Adam was chosen just because it was used in the Rainbow paper. They told it's "less sensitive to the choice of the learning rate than RMSProp".

Trained models are stored in the `checkpoins` folder and TensorBoard statistics is written into the `train` folder.

## Architecture
Only Fully Connected layers are used. Layers 2-6 are noisy linear layers, when this feature is enabled.

1. Layer #1: 37 → 128; Leaky ReLU
3. Value Layer #1: 128 → 128; Leaky ReLU
4. Value Layer #2: 128 → 1
5. Advantage Layer #1: 128 → 128; Leaky ReLU
6. Advantage Layer #2: 128 → `n_actions`;
7. Action-Value Head: Value + Advantage - AVG(Advantage)

I wasn't varying much amount of hidden units. Number of 128 units was chosen because it's a few times bigger than a maximum size of the used environmens. It worked well on all test environments and decreasing it 2-3 times less haven't done significant impact on training time.

## Running trained agent on Banana environment
First copy the saved network from the cloud to the local machine:
`wget https://github.com/vladhc/rl-udacity/raw/master/p1_navigation/checkpoints/banana-double-dueling-noisy-3306.pth`
and then run visualization:
`python play.py --env banana-vis --checkpoint banana-double-dueling-noisy-3306.pth`

This runs multiple episodes and outputs the average reward. Here is an example of running `play.py` of the trained agend on 100 episodes:

```
...
Reward #90: 15.0; Average: 14.011111111111111
Reward #91: 8.0; Average: 13.945054945054945
Reward #92: 18.0; Average: 13.98913043478261
Reward #93: 16.0; Average: 14.010752688172044
Reward #94: 15.0; Average: 14.02127659574468
Reward #95: 16.0; Average: 14.042105263157895
Reward #96: 18.0; Average: 14.083333333333334
Reward #97: 17.0; Average: 14.11340206185567
Reward #98: 0.0; Average: 13.96938775510204
Reward #99: 16.0; Average: 13.98989898989899
Reward #100: 19.0; Average: 14.04
```
Previous 90 lines were excluded for brevity. But trust me, they were much better than the last 10 episodes.

## Technical details
* Headless banana environment was used for training;
* TensorBoard is used for statistics display:
  * Episode steps, Episode Reward
  * Replay buffer size
  * Loss averaged during the episode run
  * Action value estimation: averaged during the episode run and value in the beginning of the episode
  * Epsilon values (not provided for Noisy Network)
  * Time spent on sampling the environment, optimization of the network and "all other code".
* I had an issues running multiple Banana environments on one machine. That's why training was done in the GCP using 5 `g1-small` instances. That allowed to train multiple agents concurrently with little price. Instances were without GPU - it didn't have big impact on the training time, giving 200-400 optimization steps per second. In total training ~10 banana experiments costed me about 35 euros.

## Other thoughts
* NoisyNets and Prioritirized experience replay improvements were chosen randomly, before reading the Rainbow paper. According to the paper, it would be more beneficial to implement Learning from multi-step bootstrap targets and Distributional Q-Learning.
* Training looks unpredictable. Implementing features (like NoisyNet or Dueling) showed some improvements on the CartPole and LunarLander environments, but the Banana enviroment training behaved differently each time. Out of 10 experiments only 2 can be counted as successful. Although most of experiments didn't have big variation of hyperparameters/enabled features. I've got an impression, that the training result is unpredictable and training is unstable.
* Surprisingly, NoisyNets did work. It's even more surprisingly if we'll have a look at the graph of the noise parameters of the best performing network: network was increasing it's noise and nevertheless, it was successfully training. Here is how the noise level of the advantage FC layer was changing through the training process:

![noise graph](https://github.com//vladhc/rl-udacity/raw/master/p1_navigation/imgs/noise-level.png "Noise graph")

* Surprisingly, the network with all enabled "improvements" (dueling, priority replay, etc) was often performing worse, than the network with only one improvement. And NoisyNets improvement performed much worse than simple baseline model on the CartPole environment. Although one can say, that such conclusions should be done only after averaging results of multiple experiments with the same hyperparameters, I still got an impression, that the training process is unpredictable.
* Development and conclusions can be done much faster, when the experiments can be run quickly. For example, in order to check if the improvement works as expected or not, or how the hyperparameter influences the training, one should run multiple experiments, average the results and then do the conclusions. That means, that we need intensively use possibilities of cloud platforms and use parallel computations when possible.
* Bugs are very hard to catch in such libraries. Although there is test library for testing neural networks, it covers only small part of possible bugs. That's one of the reasons why I want to keep size of the library as small, as possible. Most of the bugs were fixed just re-reading the code and re-reading the source papers. This is very tedious and time counsuming process. I haven't found solution for it so far. If you know - please let me know, I would highly appreciate.

## Future work
* Regarding the Banana Environment: try to process stack of frames instead of one frame. Currently in some situations agent "stucks" between two states trying to do perform actions "left" and "right". Adding LSTM layer and stack of frames would allow him to "remember" what was seen previously and avoid such situations. On the other hand, it could increase the learning time.
* Implement two other improvements from the Rainbow [Rainbow](https://arxiv.org/pdf/1710.02298.pdf) paper, which are considered to make the biggest impact on the training process:
  * [Distributional RL](https://arxiv.org/pdf/1707.06887.pdf)
  * Multi-step learning
* Learning to play from pixels doesn't seem to give any benefit - it justs makes the state vector bigger and making the training slower. Time should be spend on other things.
