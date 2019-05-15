import argparse
import math
import torch
import numpy as np
from collections import deque

from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from rl import QLearning, UnityEnvAdapter, Runner

from unityagents import UnityEnvironment
import gym
from gym import spaces


class SpartaWrapper(gym.Wrapper):

    def step(self, action):
        """ Pain only → reward = -1, that's why this is Sparta!!! """
        state, reward, done, debug = self.env.step(action)
        reward = -1.0 if done else 0.0
        return state, reward, done, debug


class WrapNormalizeState(gym.ObservationWrapper):

    def __init__(self, env, min_value, max_value):
        super(WrapNormalizeState, self).__init__(env)
        self.min_value = min_value
        self.max_value = max_value

    def observation(self, observation):
        x = np.array(observation)
        x -= self.min_value
        x /= self.max_value - self.min_value
        x = 2.0 * x - 1.0  # Rescale in range [-1, 1].
        return x


class WrapPyTorch(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        observation = np.array(observation)
        return observation.transpose(2, 0, 1)


def createDoneFn(steps, reward):
    rewards = deque(maxlen=steps)

    def training_done_fn(reward_acc):
        rewards.append(reward_acc)
        return np.asarray(rewards).mean() >= reward

    return training_done_fn


def createGymEnv(env_id):
    env = gym.make(env_id)

    if env_id == "CartPole-v1":
        min_vals = np.array([-2.4, -5., -math.pi/12., -math.pi*2.])
        max_vals = np.array([2.4, 5., math.pi/12., math.pi*2.])
        env = WrapNormalizeState(env, min_vals, max_vals)

    return env


def createAtariEnv(env_id):
    env = make_atari(env_id)
    env = wrap_deepmind(
            env,
            episode_life=True,
            clip_rewards=True,
            frame_stack=True,
            scale=True)
    env = WrapPyTorch(env)

    return env


def createBananaEnv():
    env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64")
    env = UnityEnvAdapter(env)

    return env


def main(**args):
    env_id = args['env']
    if env_id == "banana":
        env = createBananaEnv()
    elif env_id == "PongNoFrameskip-v4":
        env = createAtariEnv(env_id)
    else:
        env = createGymEnv(env_id)
    args['env'] = env
    print("Created {} environment".format(env_id))

    iterations = args['iterations']
    del args['iterations']
    training_steps = args['steps']
    del args['steps']
    max_episode_steps = args['max_episode_steps']
    del args['max_episode_steps']

    ql = QLearning(
            beta_decay=(iterations * training_steps),
            **args)

    runner = Runner(
            env,
            ql,
            args['sess'],
            num_iterations=iterations,
            training_steps=training_steps,
            max_episode_steps=max_episode_steps)
    runner.run_experiment()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sess")
    parser.add_argument("--env")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--noisy", action="store_true")
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--epsilon_decay", type=int, default=3000)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--max_episode_steps", type=int, default=2000)
    parser.add_argument("--target_update_freq", type=int, default=100)
    parser.add_argument("--epsilon_start", type=float, default=0.5)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)
    parser.add_argument("--hidden_units", type=int, default=128)

    parser.set_defaults(dueling=False)
    parser.set_defaults(double=False)
    parser.set_defaults(noisy=False)
    parser.set_defaults(priority=False)
    args = parser.parse_args()

    d = vars(args)
    main(**d)
