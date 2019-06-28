import time
import math
import numpy as np

import gym
from gym import spaces
from unityagents import UnityEnvironment

from rl import Statistics


unity_envs = {
        "banana":  "Banana_Linux/Banana.x86_64",
        "reachersingle": "Reacher_Linux_single/Reacher.x86_64",
        "reacher": "Reacher_Linux/Reacher.x86_64",
        "crawler": "Crawler_Linux/Crawler.x86_64",
        "tennis": "Tennis_Linux/Tennis.x86_64",
}


def create_env(env_id, count=1):
    if env_id in unity_envs:
        render = count == 1
        env = createUnityEnv(env_id, render=render)
    else:
        env = MultiGymEnv(env_id, count=count)
    print("Created {} environment. Instances: {}".format(env_id, count))
    return env


class MultiGymEnv(object):
    """ Simulates mutliple simultaneously running environments """

    def __init__(self, env_id, count):
        assert count > 0
        self.n_agents = count
        self._envs = [createGymEnv(env_id) for _ in range(count)]
        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space

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


def createUnityEnv(env_id, render=False):
    file_name = unity_envs[env_id]

    if not render:
        # Reacher_Linux/Reacher.x86_64 → Reacher_Linux_NoVis/Reacher.x86_64
        slash_idx = file_name.index("/")
        file_name = file_name[:slash_idx] + "_NoVis" + file_name[slash_idx:]

    file_name = "environments/{}".format(file_name)
    env = UnityEnvironment(file_name=file_name)
    return UnityEnvAdapter(env, env_id)


def createGymEnv(env_id):
    env = gym.make(env_id)

    if env_id == "CartPole-v1":
        min_vals = np.array([-2.4, -5., -math.pi/12., -math.pi*2.])
        max_vals = np.array([2.4, 5., math.pi/12., math.pi*2.])
        env = WrapNormalizeState(env, min_vals, max_vals)

    return env


class UnityEnvAdapter:

    def __init__(self, unity_env, name):
        self._env = unity_env
        self._name = name
        self._brain_name = self._env.brain_names[0]
        print("Unity Environment Adapter:")
        print("\tUsing brain {}".format(self._brain_name))

        # Reset the environment
        brain_info = self._env.reset(train_mode=False)[self._brain_name]

        print("\tNumber of agents: {}".format(len(brain_info.agents)))
        self.n_agents = len(brain_info.agents)


        # Number of actions
        brain = self._env.brains[self._brain_name]

        state_size = brain.vector_observation_space_size
        if self._name == "tennis":
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

        assert rewards.shape == (self.n_agents,)

        for idx in range(len(self.states)):
            env_stat = self._env_stats[idx]
            env_stat.set("steps", 1)
            env_stat.set("rewards", rewards[idx])
            if dones[idx]:
                stats.set("steps", env_stat.sum("steps"))
                stats.set("rewards", env_stat.sum("rewards"))
                stats.set("episodes", 1)
                self._env_stats[idx] = Statistics()

        # Experiment for the TennisBrain
        if self._name == "tennis":
            rewards[rewards > 0] = 10.0
            rewards[rewards < 0] = -10.0
            dones[rewards >= 0] = False

        stats.set("env_time", time.time() - t0)
        return rewards, next_states, dones, stats

    def reset(self):
        """ return state """
        self._env_stats = [Statistics() for _ in range(self.n_agents)]
        env_info = self._env.reset(train_mode=False)[self._brain_name]
        states = env_info.vector_observations
        self.states = states
        return self.states

    def render(self):
        return

    def close(self):
        return
