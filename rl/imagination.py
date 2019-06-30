from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl import TrajectoryBuffer


def train_env_model(session, state_size, action_size, epochs=10, agents_count=1):
    print("Training environment simulation model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    filename = "simulators/{}.pth".format(session)

    try:
        net = torch.load(filename)
    except FileNotFoundError:
        print("\tNo saved env model found. Creating a new one.")
        net = Net(state_size=state_size,
                action_size=action_size,
                agents_count=agents_count)

    net.to(device)
    net.train(True)

    filenames = glob("trajectories/{}-*.pth".format(session))
    filenames.sort()
    print("Using {} trajectory files for training the environment model".format(len(filenames)))
    dataset_gen = dataset_generator(filenames, agents_count, device,
            state_size=state_size, action_size=action_size)

    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    loss_fn = nn.MSELoss()

    save_model = lambda: torch.save(net, filename)

    for epoch in range(epochs):
        losses = []
        batches = dataset_gen()

        for features, labels in batches:
            optimizer.zero_grad()
            outputs = net(features)

            loss = loss_fn(outputs["state"], labels["state"]) + \
                loss_fn(outputs["reward"], labels["reward"]) + \
                loss_fn(outputs["done"], labels["done"])
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if epoch % 10 == 0:
                save_model()

        loss = np.average(losses)
        print("\t{}: {}".format(epoch, loss))

    save_model()

    return net


class Net(nn.Module):

    def __init__(self, state_size, action_size, agents_count):
        hidden_units = int(
                (state_size * agents_count + action_size * agents_count) * 1.5)
        self._state_size = state_size
        self._action_size = action_size
        self._agents_count = agents_count

        super(Net, self).__init__()

        self.middleware = nn.Sequential(
            nn.Linear(
                state_size * agents_count + action_size * agents_count,
                hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
        )
        self.head_state = nn.Linear(
                hidden_units, state_size * agents_count)
        self.head_reward = nn.Linear(
                hidden_units, agents_count)
        self.head_done = nn.Linear(
                hidden_units, agents_count)

    def forward(self, features):
        states = features["state"]
        states = states.view(-1, self._state_size * self._agents_count)

        actions = features["action"]
        actions = actions.view(-1, self._action_size * self._agents_count)

        x = torch.cat([states, actions], dim=1)
        x = self.middleware(x)

        next_state = self.head_state(x).view(-1, self._state_size, self._agents_count)
        reward = self.head_reward(x).view(-1, self._agents_count)
        done = F.sigmoid(self.head_done(x)).view(-1, self._agents_count)

        return {
            "state": next_state,
            "reward": reward,
            "done": done,
        }


def dataset_generator(
        trajectory_files, agents_count, device,
        state_size, action_size,
        batch_size=512):

    feature_states = []
    feature_actions = []

    label_rewards = []
    label_states = []
    label_dones = []

    total_records = 0

    for filename in trajectory_files:
        b = TrajectoryBuffer.load(filename)

        for traj_idx in range(0, len(b.trajectories), agents_count):
            # TODO: add generation of the first state to the dataset
            trajs = b.trajectories[traj_idx:traj_idx + agents_count]

            assert len(trajs) == agents_count

            # Make sure that all trajectories in the episode have
            # the same length
            traj_len = len(trajs[0])
            for traj in trajs:
                assert len(traj) == traj_len

            # env_idx should increase by 1
            # for idx in range(agents_count):
                # assert trajs[idx].env_idx == idx

            feature_states.append(np.stack(
                [traj.states for traj in trajs], axis=2))
            feature_actions.append(np.stack(
                [traj.actions for traj in trajs], axis=2))

            label_rewards.append(np.stack(
                [traj.rewards for traj in trajs], axis=1))
            label_states.append(np.stack(
                [traj.next_states for traj in trajs], axis=2))

            dones = np.zeros((traj_len, agents_count), np.uint8)
            for idx in range(agents_count):
                if trajs[idx].terminated:
                    dones[-1, idx] = True
            label_dones.append(dones)

        total_records += len(b)

    feature_states = np.concatenate(feature_states, axis=0)
    feature_actions = np.concatenate(feature_actions, axis=0)

    label_rewards = np.concatenate(label_rewards, axis=0)
    label_states = np.concatenate(label_states, axis=0)
    label_dones = np.concatenate(label_dones, axis=0)

    total_records = int(total_records / agents_count)

    print(total_records, action_size, agents_count)

    assert feature_states.shape == (
            total_records, state_size, agents_count), feature_states.shape
    assert feature_actions.shape == (
            total_records, action_size, agents_count), feature_actions.shape
    assert label_states.shape == feature_states.shape, label_states.shape
    assert label_rewards.shape == (
            total_records, agents_count), label_rewards.shape
    assert label_dones.shape == (
            total_records, agents_count), label_dones.shape

    def _dataset_iterator():
        shuffled_idx = np.random.choice(
                total_records,
                size=total_records,
                replace=False)
        batches = int(total_records / batch_size)

        for batch_idx in range(batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(feature_states))

            idx = shuffled_idx[start:end]
            states = feature_states[idx]
            states = torch.from_numpy(states).float().to(device)

            actions = feature_actions[idx]
            actions = torch.from_numpy(actions).float().to(device)

            next_states = label_states[idx]
            next_states = torch.from_numpy(next_states).float().to(device)

            rewards = label_rewards[idx]
            rewards = torch.from_numpy(rewards).float().to(device)

            dones = label_dones[idx]
            dones = dones.astype(np.uint8)
            dones = torch.from_numpy(dones).float().to(device)

            yield ({
                "state": states,
                "action": actions,
            }, {
                "state": next_states,
                "reward": rewards,
                "done": dones,
            })

    return _dataset_iterator
