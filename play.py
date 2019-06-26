import argparse
import torch
import time
import os
from rl import create_env, GreedyPolicy
import numpy as np
from gym import spaces


def main(checkpoint):
    filename = os.path.basename(checkpoint)
    s = filename.split('-')

    # Derive environment ID from the checkpoint filename
    file_prefix = s[0]
    env_id = {
            "pole": "CartPole-v1",
            "lunarcont": "LunarLanderContinuous-v2",
            "lunar": "LunarLander-v2",
            "carcont": "MountainCarContinuous-v0",
            "pendulum": "Pendulum-v0",
            "reacher": "reacher",
            "crawler": "crawler",
            "banana": "banana",
            }[file_prefix]
    s = s[1:]

    env = create_env(env_id)
    sample_action = sample_action_fn(checkpoint, env.action_space)

    rewards = []

    try:
        while True:
            reward = play_episode(env, sample_action)
            rewards.append(reward)
            print("Reward #{}: {}; Average: {}".format(
                len(rewards), rewards[-1], np.average(rewards)))
    except KeyboardInterrupt:
        env.close()
        return
    env.close()


def play_episode(env, sample_action):
    env.reset()
    reward_acc = 0.0

    while True:
        actions = sample_action(env.states)
        rewards, _, dones, _ = env.step(actions)
        env.render()
        time.sleep(0.01)
        reward_acc += np.average(rewards)
        if dones.any():
            break
    return reward_acc


def sample_action_fn(checkpoint, action_space):
    net = torch.load(checkpoint, map_location="cpu")
    net.train(False)

    is_continous = isinstance(action_space, spaces.Box)

    policy = GreedyPolicy()

    def _qlearning(states):
        states_tensor = torch.from_numpy(states).float()
        q_values = net(states_tensor)
        return policy.get_action(q_values.detach().cpu().numpy())

    def _ppo(states):
        states_tensor = torch.from_numpy(states).float()
        if is_continous:
            batch_size = len(states)
            action_shape = (batch_size, ) + action_space.shape
            actions_mu, actions_var, _ = net(states_tensor)
            actions_arr = []
            for action_idx in range(action_space.shape[0]):
                action_mu = actions_mu[:, action_idx]
                action_var = actions_var[:, action_idx]
                assert action_mu.shape == (batch_size,), action_mu.shape
                assert action_var.shape == (batch_size,), action_var.shape
                dist = torch.distributions.Normal(
                        action_mu,
                        action_var)
                sub_actions = dist.sample()
                actions_arr.append(sub_actions)
            actions = torch.stack(actions_arr, dim=1)
            # Each action can consist of multiple sub-actions
            assert actions.shape == action_shape, actions.shape
        else:
            with torch.no_grad():
                action_logits, _, _ = net(states_tensor)
                dist = torch.distributions.categorical.Categorical(
                        logits=action_logits)
                actions = dist.sample()
                assert actions.shape == (batch_size,), actions.shape
        return actions.detach().cpu().numpy()

    # Derive agent from the checkpoint filename
    filename = os.path.basename(checkpoint)
    for s in filename.split('-'):
        if s == "ppo":
            return _ppo

    return _qlearning


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    main(args.checkpoint)
