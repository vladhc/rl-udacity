import time
import math
import numpy as np
from collections import namedtuple
from multiprocessing import Process, Pipe, set_start_method

import gym
from gym import spaces
from unityagents import UnityEnvironment

from rl import Statistics

# Common code for both Unity and OpenAI environments

def create_env(env_id, count=1):
    set_start_method('spawn')
    assert count > 0

    if env_id in unity_envs:
        render = count == 1
        fork = count > 1
        if fork:
            create_env_fn = lambda: ForkedUnityEnv(env_id, render=render)
        else:
            create_env_fn = lambda: _run_unity_env(
                    env_id,
                    render=render,
                    worker_id=0)
    else:
        create_env_fn = lambda: OpenAIAdapter(env_id)

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

        step_promises = []
        for env_idx, env in enumerate(self._envs):
            count = env.n_envs * env.n_agents
            start = env_idx * count
            end = start + count
            env_actions = actions[start:end]
            step_promises.append(env.step(env_actions))

        reset_states = []
        for env_idx, step_promise in enumerate(step_promises):
            env_next_states, env_rewards, env_dones = step_promise()

            env = self._envs[env_idx]
            env_states = env_next_states
            env_stat = self._env_stats[env]
            env_stat.set("steps", 1)
            env_stat.set("rewards", sum(env_rewards))
            if env_dones.any():
                stats.set("steps", env_stat.sum("steps"))
                avg_reward = env_stat.sum("rewards") / (
                        env.n_envs * env.n_agents)
                stats.set("rewards", avg_reward)
                stats.set("episodes", 1)
                self._env_stats[env] = Statistics()
                reset_states.append((env_idx, env.reset()))

            next_states.append(env_next_states)
            states.append(env_states)
            rewards.append(env_rewards)
            dones.append(env_dones)

        for env_idx, step_promise in reset_states:
            states[env_idx] = step_promise()

        rewards = np.concatenate(rewards, axis=0)
        dones = np.concatenate(dones, axis=0)
        next_states = np.concatenate(next_states, axis=0)
        self.states = np.concatenate(states, axis=0)

        stats.set("env_time", time.time() - t0)

        return rewards, next_states, dones, stats

    def reset(self):
        self._env_stats = {env: Statistics() for env in self._envs}
        step_promises = [env.reset() for env in self._envs]
        states = [step_promise() for step_promise in step_promises]
        self.states = np.concatenate(states, axis=0)

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

    def step(self, actions):
        assert len(actions) == 1
        state, reward, done, _ = self._env.step(actions[0])
        states = np.expand_dims(state, axis=0)
        rewards = np.expand_dims(reward, axis=0)
        dones = np.expand_dims(done, axis=0)
        return lambda: (states, rewards, dones)

    def reset(self):
        state = self._env.reset()
        states = np.expand_dims(state, axis=0)
        return lambda: states

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


class ForkedUnityEnv:

    last_unity_worker_id = 0

    def __init__(self, env_id, render=False):
        self._config = unity_envs[env_id]
        traj_pipe_in, traj_pipe_out = Pipe(duplex=False)
        action_pipe_in, action_pipe_out = Pipe(duplex=False)

        self._traj_pipe = traj_pipe_in
        self._action_pipe = action_pipe_out

        p = Process(
                target=_run_forked_unity_env,
                args=(
                    env_id,
                    action_pipe_in,
                    traj_pipe_out,
                    render,
                    ForkedUnityEnv.last_unity_worker_id))
        p.start()
        ForkedUnityEnv.last_unity_worker_id += 1

        env_info = self._traj_pipe.recv()
        self.observation_space = env_info["observation_space"]
        self.action_space = env_info["action_space"]
        self.n_agents = env_info["n_agents"]
        self.n_envs = env_info["n_envs"]

    def step(self, actions):
        self._action_pipe.send(actions)
        return self._traj_pipe.recv

    def reset(self):
        self._action_pipe.send("RESET")
        return self._traj_pipe.recv

    def render(self):
        self._action_pipe.send("RENDER")

    def close(self):
        self._action_pipe.send("CLOSE")


def _run_forked_unity_env(env_id, action_pipe, traj_pipe, render, worker_id):
    env = _run_unity_env(env_id, worker_id=worker_id, render=render)

    traj_pipe.send({
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "n_agents": env.n_agents,
        "n_envs": env.n_envs,
    })

    while True:
        try:
            action = action_pipe.recv()
            if action == "RESET":
                state_promise = env.reset()
                state = state_promise()
                traj_pipe.send(state)
            elif action == "CLOSE":
                env.close()
                return
            elif action == "RENDER":
                env.render()
            else:
                step_promise = env.step(action)
                step = step_promise()
                traj_pipe.send(step)
        except KeyboardInterrupt:
            pass


def _run_unity_env(env_id, worker_id, render=False):
    config = unity_envs[env_id]
    return UnityEnvAdapter(config, render=render, worker_id=worker_id)


class UnityEnvAdapter:

    def __init__(self, config, render, worker_id):
        self._config = config

        file_name = config.path
        if not render:
            file_name = config.path_novis
        file_name = "environments/{}".format(file_name)

        self._env = UnityEnvironment(file_name=file_name, worker_id=worker_id)
        self._brain_name = self._env.brain_names[0]
        print("Unity Environment Adapter:")
        print("\tUsing brain {}".format(self._brain_name))

        # Number of actions
        brain = self._env.brains[self._brain_name]

        observation_shape = (brain.vector_observation_space_size,)
        if config.id == "tennis":
            # Fix for the TennisBrain
            observation_shape = (3, brain.vector_observation_space_size)

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
                    shape=observation_shape)
        else:
            self.observation_space = spaces.Discrete(
                    brain.vector_observation_space_size)
        print("\tState shape:", self.observation_space.shape)

    def step(self, actions):
        """ return next_state, action, reward, None """
        brain_info = self._env.step(actions)[self._brain_name]
        next_states = self._check_states(brain_info.vector_observations)
        rewards = brain_info.rewards
        rewards = np.asarray(rewards)
        dones = np.asarray(brain_info.local_done)

        assert rewards.shape == (self._batch_size,)

        return lambda: (next_states, rewards, dones)

    def reset(self):
        """ return state """
        env_info = self._env.reset(train_mode=False)[self._brain_name]
        states = self._check_states(env_info.vector_observations)
        if self._config.id == "tennis":
            noop = np.zeros((2, 2))
            self.step(noop)()
            step_promise = self.step(noop)
            states, _, _ = step_promise()
        return lambda: states

    def _check_states(self, states):
        if self._config.id == "tennis":
            # Fix for the TennisBrain
            states = states.reshape(2, 3, 8)
        return states

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
