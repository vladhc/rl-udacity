import argparse
import torch
import numpy as np
from collections import deque

from rl import QLearning, UnityEnvAdapter

from unityagents import UnityEnvironment
import gym
from gym import spaces


class SpartaWrapper(gym.Wrapper):

    def step(self, action):
        """ Pain only → reward = -1, that's why this is Sparta!!! """
        state, reward, done, debug = self.env.step(action)
        reward = -1.0 if done else 0.0
        return state, reward, done, debug


def createDoneFn(steps, reward):
    rewards = deque(maxlen=steps)

    def training_done_fn(reward_acc):
        rewards.append(reward_acc)
        return np.asarray(rewards).mean() >= reward

    return training_done_fn


def createGymEnv(env_id):
    env = gym.make(env_id)

    training_done_fn = lambda x: False
    if env_id == "CartPole-v1":
        env = SpartaWrapper(env)
        training_done_fn = createDoneFn(100, 95.0)
    if env_id == "LunarLander-v2":
        training_done_fn = createDoneFn(100, 200.0)

    return env, training_done_fn


def createBananaEnv():
    env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64")
    env = UnityEnvAdapter(env)

    training_done_fn = createDoneFn(100, 14.0)

    print("Created Banana environment")
    return env, training_done_fn


def main(**args):
    if args['env'] == "banana":
        env, done_fn = createBananaEnv()
    else:
        env, done_fn = createGymEnv(args['env'])
    args['env'] = env

    steps = args['steps']
    del args['steps']

    ql = QLearning(
        training_done_fn=done_fn,
        **args)
    ql.train(steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sess")
    parser.add_argument("--env")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--epsilon_decay", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--max_episode_steps", type=int, default=2000)
    parser.add_argument("--target_update_freq", type=int, default=100)
    parser.add_argument("--epsilon_start", type=float, default=0.5)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)

    parser.set_defaults(dueling=False)
    parser.set_defaults(double=False)
    parser.set_defaults(noisy=False)
    parser.set_defaults(priority=False)
    args = parser.parse_args()

    d = vars(args)
    main(**d)
