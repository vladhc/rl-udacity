import torch
import numpy as np

from rl import QLearning, UnityEnvAdapter

from unityagents import UnityEnvironment
import gym

class SpartaWrapper(gym.Wrapper):
    """ Pain only → reward = -1, that's why this is Sparta!!! """
    def step(self, action):
        state, reward, done, debug = self.env.step(action)
        reward = -1.0 if done else 0.0
        return state, reward, done, debug


def createCartPoleEnv():
    env = gym.make("CartPole-v1")
    env = SpartaWrapper(env)
    env.state_size = 4
    env.action_size = 2
    return env

def createBananaEnv():
    env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64")
    env = UnityEnvAdapter(env)
    return env

def main():
    env = createCartPoleEnv()

    # Reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    ql = QLearning(
        env,
        observation_size=env.state_size,
        action_size=env.action_size,
        max_episode_steps=2000,
        target_update_freq=100,
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay=3000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.001,
        replay_buffer_size=10000)
    ql.train(1500)

if __name__ == '__main__':
    main()
