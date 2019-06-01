import gym
from unityagents import UnityEnvironment
import numpy as np
import math

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from gym import spaces


def create_env(env_id):
    if env_id.startswith("banana") or env_id.startswith("reacher"):
        env = createUnityEnv(env_id)
    elif env_id == "PongNoFrameskip-v4":
        env = createAtariEnv(env_id)
    else:
        env = createGymEnv(env_id)
    print("Created {} environment".format(env_id))
    return env


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
        self.single_agent = len(brain_info.agents) == 1

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

    def step(self, action):
        """ return next_state, action, reward, None """
        brain_info = self._env.step(action)[self._brain_name]
        next_states = brain_info.vector_observations
        rewards = brain_info.rewards
        dones = brain_info.local_done
        if self.single_agent:
            return next_states[0], rewards[0], dones[0], None
        return next_states, rewards, dones, None

    def reset(self):
        """ return state """
        env_info = self._env.reset(train_mode=False)[self._brain_name]
        states = env_info.vector_observations
        if self.single_agent:
            return states[0]
        return states

    def render(self):
        return
