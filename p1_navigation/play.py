import argparse
import torch
from rl import create_env, GreedyPolicy, Statistics
import numpy as np


def main(env_id, checkpoint):
    try:
        env = create_env(env_id)
        q_net = torch.load(checkpoint, map_location='cpu')
        q_net.train(False)

        rewards = []
        while True:
            reward = play_episode(env, q_net)
            rewards.append(reward)
            print("Reward #{}: {}; Average: {}".format(
                len(rewards), rewards[-1], np.average(rewards)))
    except KeyboardInterrupt:
        env.close()
        return
    env.close()


def play_episode(env, q_net):
    policy = GreedyPolicy()
    env.reset()
    reward_acc = 0.0

    while True:
        state_tensor = torch.from_numpy(env.states).float()
        with torch.no_grad():
            q_values = q_net(state_tensor)
        actions = policy.get_action(q_values.cpu().numpy())
        rewards, _, dones, _ = env.step(actions)
        env.render()
        reward_acc += np.average(rewards)
        if dones.any():
            break
    return reward_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--env")

    args = parser.parse_args()

    main(args.env, args.checkpoint)
