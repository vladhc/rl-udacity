import argparse
import torch
import torch.nn as nn
import time
import os
from rl import create_env, GreedyPolicy, Statistics
import numpy as np
from gym import spaces


def main(checkpoint, debug=False):
    filename = os.path.basename(checkpoint)
    s = filename.split('-')

    # Create Environment
    # Derive environment ID from the checkpoint filename
    file_prefix = s[0]
    openai_env_ids = {
            "pole": "CartPole-v1",
            "lunarcont": "LunarLanderContinuous-v2",
            "lunar": "LunarLander-v2",
            "carcont": "MountainCarContinuous-v0",
            "pendulum": "Pendulum-v0",
    }
    if file_prefix in openai_env_ids:
        env_id = openai_env_ids[file_prefix]
    else:
        env_id = file_prefix

    s = s[1:]

    env = create_env(env_id)

    # Create agent
    sample_action = sample_action_fn(checkpoint, env.action_space)

    stats = Statistics()

    try:
        while True:
            episode_stats = play_episode(env, sample_action, debug=debug)
            stats.set_all(episode_stats)
            print(("Episode #{}: {:.2f}; Average Reward: {:.2f}; " +
                  "Episode length: {}; Average episode length: {:.1f}").format(
                      stats.sum("episodes"),
                      episode_stats.avg("rewards"),
                      stats.avg("rewards"),
                      int(episode_stats.avg("steps")),
                      stats.avg("steps")))
    except KeyboardInterrupt:
        env.close()
        return
    env.close()


def play_episode(env, sample_action, debug=False):
    env.reset()

    while True:
        states = env.states
        actions = sample_action(env.states)
        rewards, _, dones, stats = env.step(actions)
        if debug:
            print("states:")
            print(states)
            print("actions:", actions)
            print("rewards:", rewards)
            if dones.any():
                print("term:", dones)
        env.render()
        if debug:
            input("Press for the next step...")
        else:
            time.sleep(0.02)
        if dones.any():
            break
    return stats


def sample_action_fn(checkpoint, action_space):
    props = torch.load(checkpoint, map_location="cpu")

    if isinstance(props, nn.Module):
        props.train(False)

    is_continous = isinstance(action_space, spaces.Box)

    policy = GreedyPolicy()

    def _qlearning(states):
        states_tensor = torch.from_numpy(states).float()
        q_values = net(states_tensor)
        return policy.get_action(q_values.detach().cpu().numpy())

    def _ppo(states, props):
        net = props["net"]
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

    def _multippo(states):
        actions = []
        for idx in range(len(states)):
            key = "agent-{}".format(idx)
            agent = props[key]
            agent_states = np.expand_dims(states[idx], axis=0)
            action = _ppo(states=agent_states, props=agent)[0]
            actions.append(action)
        actions = np.asarray(actions)
        batch_size = len(states)
        actions_shape = (batch_size,) + action_space.shape
        assert actions.shape == actions_shape, actions.shape
        return actions

    # Derive agent from the checkpoint filename
    filename = os.path.basename(checkpoint)
    for s in filename.split('-'):
        if s == "ppo":
            return lambda states: _ppo(states, props)
        elif s == "multippo":
            return _multippo

    return _qlearning


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(debug=False)
    args = parser.parse_args()
    main(args.checkpoint, debug=args.debug)
