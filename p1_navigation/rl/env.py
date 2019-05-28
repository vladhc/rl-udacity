import gym
from gym import spaces
from unityagents import UnityEnvironment
import numpy as np
import math

from baselines.common.atari_wrappers import make_atari, wrap_deepmind


def create_env(env_id):
    if env_id.startswith("banana"):
        env = createBananaEnv(render=env_id.endswith("-vis"))
    elif env_id == "PongNoFrameskip-v4":
        env = createAtariEnv(env_id)
    else:
        env = createGymEnv(env_id)
    print("Created {} environment".format(env_id))
    return env


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


def createBananaEnv(render):
    f = "./Banana_Linux/Banana.x86_64" if render \
            else "./Banana_Linux_NoVis/Banana.x86_64"
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

        # reset the environment
        env_info = self._env.reset(train_mode=False)[self._brain_name]

        # number of actions
        brain = self._env.brains[self._brain_name]
        self.action_size = brain.vector_action_space_size
        print('Number of actions:', self.action_size)
        # examine the state space
        state = env_info.vector_observations[0]
        self.state_size = len(state)
        print('State size:', self.state_size)

    def step(self, action):
        """ return next_state, action, reward, None """
        env_info = self._env.step(action)[self._brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done, None

    def reset(self):
        """ return state """
        env_info = self._env.reset(train_mode=False)[self._brain_name]
        state = env_info.vector_observations[0]
        return state

    def render(self):
        return
