import gym
from unityagents import UnityEnvironment
import numpy as np
import math

from gym import spaces


def create_env(env_id, count=1):
    if env_id.startswith("banana") or env_id.startswith("reacher"):
        env = createUnityEnv(env_id)
    else:
        env = MultiGymEnv(env_id, count=count)
    print("Created {} environment".format(env_id))
    return env


class MultiGymEnv(object):

    def __init__(self, env_id, count):
        assert count > 0
        self.n_agents = count
        self._envs = [createGymEnv(env_id) for _ in range(count)]
        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space

    def step(self, actions):
        next_states = []
        rewards = []
        dones = []

        envs = [env for env in self._envs if not self._skip[env]]

        for action, env in zip(actions, envs):
            next_state, reward, done, _ = env.step(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            self._skip[env] = done

        states = self.states
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        next_states = np.asarray(next_states)
        self.states = next_states[~dones]
        return states, actions, rewards, next_states, dones

    def reset(self):
        self._skip = {env: False for env in self._envs}
        states = [env.reset() for env in self._envs]
        self.episodes_count = 0
        self.states = np.asarray(states)
        return self.states

    def render(self):
        if len(self._env) == 1:
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


def createUnityEnv(env_id):
    if env_id.startswith("banana"):
        render = env_id.endswith("-vis")
        f = "Banana_Linux/Banana.x86_64" if render \
            else "Banana_Linux_NoVis/Banana.x86_64"
    elif env_id.startswith("reacher"):
        single = env_id.endswith("-single")
        f = "Reacher_Linux_single/Reacher.x86_64" if single \
            else "Reacher_Linux/Reacher.x86_64"
    f = "environments/{}".format(f)
    env = UnityEnvironment(file_name=f)
    return UnityEnvAdapter(env)


def createGymEnv(env_id):
    env = gym.make(env_id)

    if env_id == "CartPole-v1":
        min_vals = np.array([-2.4, -5., -math.pi/12., -math.pi*2.])
        max_vals = np.array([2.4, 5., math.pi/12., math.pi*2.])
        env = WrapNormalizeState(env, min_vals, max_vals)

    return env


class UnityEnvAdapter:

    def __init__(self, unity_env):
        self._env = unity_env
        self._brain_name = self._env.brain_names[0]
        print("Unity Environment Adapter:")
        print("\tUsing brain {}".format(self._brain_name))

        # Reset the environment
        brain_info = self._env.reset(train_mode=False)[self._brain_name]

        print("\tNumber of agents: {}".format(len(brain_info.agents)))
        self.n_agents = len(brain_info.agents)

        # Number of actions
        brain = self._env.brains[self._brain_name]
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
                    shape=(brain.vector_observation_space_size,))
        else:
            self.observation_space = spaces.Discrete(
                    brain.vector_observation_space_size)
        print("\tState shape:", self.observation_space.shape)

    def step(self, actions):
        """ return next_state, action, reward, None """
        brain_info = self._env.step(actions)[self._brain_name]
        next_states = brain_info.vector_observations
        rewards = brain_info.rewards
        dones = brain_info.local_done
        states = self.states
        self.states = next_states
        return states, actions, rewards, next_states, dones

    def reset(self):
        """ return state """
        env_info = self._env.reset(train_mode=False)[self._brain_name]
        states = env_info.vector_observations
        self.states = states
        return self.states

    def render(self):
        return
