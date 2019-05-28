import argparse
import torch
from rl import create_env, GreedyPolicy
import numpy as np


def main(env_id, checkpoint):
    env = create_env(env_id)
    q_net = torch.load(checkpoint, map_location='cpu')
    q_net.train(False)

    rewards = []
    while True:
        reward = play_episode(env, q_net)
        rewards.append(reward)
        print("Reward #{}: {}; Average: {}".format(
            len(rewards), rewards[-1], np.average(rewards)))


def play_episode(env, q_net):
    policy = GreedyPolicy()
    state = env.reset()
    reward_acc = 0.0
    while True:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        q_values = q_net(state_tensor)
        action = policy.get_action(q_values)
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state
        reward_acc += reward
        if done:
            break
    return reward_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--env")
    parser.add_argument("--iteration", type=int, default=100)

    args = parser.parse_args()

    main(args.env, args.checkpoint)
