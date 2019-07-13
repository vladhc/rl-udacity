import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl import Trajectory


class Reinforce:

    def __init__(
            self,
            action_space,
            observation_space,
            gamma=0.99,
            baseline=False,
            baseline_learning_rate=0.001,
            learning_rate=0.001):

        print("REINFORCE agent:")

        self._gamma = gamma
        print("\tReward discount (gamma): {}".format(self._gamma))

        self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self._baseline = baseline

        if self._baseline:
            self._baseline_net = PolicyBaselineNet(
                observation_space.shape,
                action_space.n)
            self._baseline_net.to(self._device)

        self._net = PolicyBaselineNet(
                observation_space.shape,
                action_space.n)
        self._net.to(self._device)

        # Optimizer and loss
        self._loss_fn = nn.MSELoss(reduce=False)
        self._optimizer = optim.Adam(
                self._net.parameters(),
                lr=learning_rate)
        print("\tLearning rate: {}".format(learning_rate))

        if self._baseline:
            self._baseline_optimizer = optim.Adam(
                            self._baseline_net.parameters(),
                            lr=baseline_learning_rate)
            print("\tLearning rate for baseline network: {}".format(
                    baseline_learning_rate))

        # Trajectory, which will be used for training
        self.create_trajectory = lambda: Trajectory(
                capacity=1000,
                observation_shape=observation_space.shape,
                action_type=np.int32,
                action_shape=action_space.shape,
                env_idx=0)
        self._trajectory = self.create_trajectory()

    def save(self):
        return {
            "net": self._net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load(self, props):
        self._net.load_state_dict(props["net"])
        self._optimizer.load_state_dict(props["optimizer"])

    def step(self, states):
        states_tensor = torch.from_numpy(states).float()
        self._net.train(False)
        with torch.no_grad():

            action_logits = self._net(states_tensor)
            action_logits = action_logits.double()
            action_probs = torch.nn.Softmax(dim=1)(action_logits)
            action_probs = action_probs.detach()

        action = np.random.choice(
                len(action_probs[0]),
                p=action_probs[0])
        return [action]

    def transitions(self, states, actions, rewards, next_states, term):
        self._trajectory.push(
                states[0],
                actions[0],
                rewards[0],
                next_states[0],
                term[0])
        if self._trajectory.done():
            self._optimize()
            self._trajectory = self.create_trajectory()

    def _optimize(self):
        self._net.train(True)

        traj = self._trajectory
        batch_size = len(traj)

        # Action Credit assignement
        gs = np.zeros(batch_size, dtype=np.float16)
        for step, reward in enumerate(traj.rewards):
            g = 0.0
            for k in range(step, len(traj)):
                g += np.power(self._gamma, k - step) * traj.rewards[k]
            gs[step] = g
        gs = torch.from_numpy(gs).float()
        gs = torch.unsqueeze(gs, dim=1)

        states = torch.from_numpy(traj.states).float()
        actions = torch.from_numpy(traj.actions).long()
        actions = torch.unsqueeze(actions, dim=1)

        assert actions.shape == (batch_size, 1), actions.shape

        action_logits = self._net(states)

        log_softmax = torch.nn.LogSoftmax(dim=1)
        action_log_probs = log_softmax(action_logits)
        action_log_probs = action_log_probs.gather(dim=1, index=actions)

        # -1 here because optimizer is going to *minimize* the
        # loss. If we were going to update the weights manually,
        # (without optimizer) we would remove the -1.
        loss = -1 * gs * action_log_probs
        assert loss.shape == (batch_size, 1), loss.shape
        loss = loss.mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._trajectory = self.create_trajectory()

    def episodes_end(self):
        self._trajectory = self.create_trajectory()


class PolicyBaselineNet(nn.Module):

    def __init__(
            self,
            observation_size,
            action_size):
        super(PolicyBaselineNet, self).__init__()

        hidden_units = 32

        self.base = nn.Sequential(
                nn.Linear(observation_size[0], hidden_units),
                nn.ReLU(),
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU())

        self.action_head = nn.Linear(hidden_units, action_size)

    def forward(self, x):
        x = self.base(x)
        action_logits = self.action_head(x)
        return action_logits
