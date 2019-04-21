import argparse
import torch
import numpy as np

from rl import QLearning, UnityEnvAdapter

from unityagents import UnityEnvironment
import gym


class SpartaWrapper(gym.Wrapper):

    def step(self, action):
        """ Pain only → reward = -1, that's why this is Sparta!!! """
        state, reward, done, debug = self.env.step(action)
        reward = -1.0 if done else 0.0
        return state, reward, done, debug


def createGymEnv(env_id):
    env = gym.make(env_id)
    if env_id == "CartPole-v1":
        env = SpartaWrapper(env)
    return env


def createBananaEnv():
    env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64")
    env = UnityEnvAdapter(env)
    print("Created Banana environment")
    return env


def main(session, env_id, dueling, double):
    if env_id == "banana":
        env = createBananaEnv()
    else:
        env = createGymEnv(env_id)

    # Reproducibility
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    ql = QLearning(
        env,
        session,
        dueling=dueling,
        double=double,
        max_episode_steps=2000,
        target_update_freq=100,
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay=3000,
        batch_size=64,
        gamma=0.99,
        learning_rate=0.001,
        replay_buffer_size=10000)
    ql.train(1800)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sess")
    parser.add_argument("--env")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--double", action="store_true")
    parser.set_defaults(dueling=False)
    args = parser.parse_args()
    main(args.sess, args.env, args.dueling, args.double)
