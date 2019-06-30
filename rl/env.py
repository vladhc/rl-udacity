import time
import math
import numpy as np
from collections import namedtuple

import gym
from gym import spaces
from unityagents import UnityEnvironment

from rl import Statistics

# Common code for both Unity and OpenAI environments

def create_env(env_id, count=1):
    assert count > 0

    if env_id in unity_envs:
        config = unity_envs[env_id]
        render = count == 1
        create_env_fn = lambda: UnityEnvAdapter(config, render=render)
    else:
        create_env_fn = lambda: OpenAIAdapter(env_id)

    if count == 1:
        env = create_env_fn()
    else:
        env = MultiEnv(create_env_fn, count=count)

    print("Created {} environment. Instances: {}".format(env_id, count))
    return env


class MultiEnv(object):
    """ Simulates mutliple simultaneously running environments """

    def __init__(self, create_env_fn, count):
        assert count > 0
        self._envs = [create_env_fn() for _ in range(count)]
        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space
        self.n_agents = self._envs[0].n_agents

    @property
    def n_envs(self):
        return sum([env.n_envs for env in self._envs])

    def step(self, actions):
        stats = Statistics()

        t0 = time.time()
        next_states = []
        states = []
        rewards = []
        dones = []
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()

        for action, env in zip(actions, self._envs):
            next_state, reward, done, _ = env.step(action)

            state = next_state
            env_stat = self._env_stats[env]
            env_stat.set("steps", 1)
            env_stat.set("rewards", reward)
            if done:
                stats.set("steps", env_stat.sum("steps"))
                stats.set("rewards", env_stat.sum("rewards"))
                stats.set("episodes", 1)
                self._env_stats[env] = Statistics()
                state = env.reset()

            next_states.append(next_state)
            states.append(state)
            rewards.append(reward)
            dones.append(done)

        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        next_states = np.asarray(next_states)
        self.states = np.asarray(states)

        stats.set("env_time", time.time() - t0)

        return rewards, next_states, dones, stats

    def reset(self):
        self._env_stats = {env: Statistics() for env in self._envs}
        states = [env.reset() for env in self._envs]
        self.states = np.asarray(states)

    def render(self):
        if len(self._envs) == 1:
            self._envs[0].render()

    def close(self):
        [env.close() for env in self._envs]


# OpenAI Environments


class OpenAIAdapter:

    def __init__(self, env_id):
        env = gym.make(env_id)

        if env_id == "CartPole-v1":
            min_vals = np.array([-2.4, -5., -math.pi/12., -math.pi*2.])
            max_vals = np.array([2.4, 5., math.pi/12., math.pi*2.])
            env = WrapNormalizeState(env, min_vals, max_vals)

        self._env = env

        self.n_envs = 1
        self.n_agents = 1
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def close(self):
        self._env.close()

    def render(self):
        self._env.render()


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


# Unity Environments


UnityEnvConfig = namedtuple("UnityEnvConfig", [
    "id", "path", "path_novis", "n_agents", "n_envs"])

unity_env_list = {
        UnityEnvConfig(
            id="banana",
            path="Banana_Linux/Banana.x86_64",
            path_novis="Banana_Linux_NoVis/Banana.x86_64",
            n_agents=1,
            n_envs=1,
        ),
        UnityEnvConfig(
            id="reachersingle",
            path="Reacher_Linux_single/Reacher.x86_64",
            path_novis="Reacher_Linux_single_NoVis/Reacher.x86_64",
            n_agents=1,
            n_envs=1,
        ),
        UnityEnvConfig(
            id="reacher",
            path="Reacher_Linux/Reacher.x86_64",
            path_novis="Reacher_Linux_NoVis/Reacher.x86_64",
            n_agents=1,
            n_envs=20,
        ),
        UnityEnvConfig(
            id="crawler",
            path="Crawler_Linux/Crawler.x86_64",
            path_novis="Crawler_Linux_NoVis/Crawler.x86_64",
            n_agents=1,
            n_envs=12,
        ),
        UnityEnvConfig(
            id="tennis",
            path="Tennis_Linux/Tennis.x86_64",
            path_novis="Tennis_Linux_NoVis/Tennis.x86_64",
            n_agents=2,
            n_envs=1,
        ),
}

unity_envs = {env.id: env for env in unity_env_list}


class UnityEnvAdapter:

    def __init__(self, config, render):
        self._config = config

        file_name = config.path
        if not render:
            file_name = config.path_novis
        file_name = "environments/{}".format(file_name)

        self._env = UnityEnvironment(file_name=file_name)
        self._brain_name = self._env.brain_names[0]
        print("Unity Environment Adapter:")
        print("\tUsing brain {}".format(self._brain_name))

        # Number of actions
        brain = self._env.brains[self._brain_name]

        state_size = brain.vector_observation_space_size
        if config.id == "tennis":
            state_size = 24  # Bugfix for the TennisBrain

        if brain.vector_action_space_type == 'continuous':
            self.action_space = spaces.Box(
                        low=-1.0, high=1.0,
                        shape=(brain.vector_action_space_size,)
                    )
        else:
            self.action_space = spaces.Discrete(brain.vector_action_space_size)
        print("\tAction space: {}".format(self.action_space))

        # Examine the state space
        if brain.vector_observation_space_type == 'continuous':
            print("\tUsing arbitrary 'high' and 'low' values for the " +
                    "observation space.")
            self.observation_space = spaces.Box(
                    low=-100.0, high=100.0,
                    shape=(state_size,))
        else:
            self.observation_space = spaces.Discrete(
                    brain.vector_observation_space_size)
        print("\tState shape:", self.observation_space.shape)

    def step(self, actions):
        """ return next_state, action, reward, None """
        stats = Statistics()
        t0 = time.time()

        brain_info = self._env.step(actions)[self._brain_name]
        next_states = brain_info.vector_observations
        rewards = brain_info.rewards
        rewards = np.asarray(rewards)
        dones = np.asarray(brain_info.local_done)
        self.states = next_states

        assert rewards.shape == (self._batch_size,)

        for idx in range(self._batch_size):
            env_stat = self._env_stats[idx]
            env_stat.set("steps", 1)
            env_stat.set("rewards", rewards[idx])
            if dones[idx]:
                stats.set("steps", env_stat.sum("steps"))
                stats.set("rewards", env_stat.sum("rewards"))
                stats.set("episodes", 1)
                self._env_stats[idx] = Statistics()

        stats.set("env_time", time.time() - t0)
        return rewards, next_states, dones, stats

    def reset(self):
        """ return state """
        self._env_stats = [Statistics() for _ in range(self._batch_size)]
        env_info = self._env.reset(train_mode=False)[self._brain_name]
        states = env_info.vector_observations
        self.states = states

    def render(self):
        return

    def close(self):
        return

    @property
    def n_envs(self):
        return self._config.n_envs

    @property
    def n_agents(self):
        return self._config.n_agents

    @property
    def _batch_size(self):
        return self.n_envs * self.n_agents
