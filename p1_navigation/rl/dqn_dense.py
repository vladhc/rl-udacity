import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from rl import NoisyLinear


def linearClass(noisy):
    if noisy:
        return NoisyLinear
    else:
        return nn.Linear


def _log_noisy(log_fn, tag, layer):
    try:
        weight = layer.sigma_weight.data.numpy()
        log_fn('noise_{}_weights'.format(tag),
                np.average(np.abs(weight)))
    except AttributeError:
        pass
    try:
        bias = layer.sigma_bias.data.numpy()
        log_fn('noise_{}_bias'.format(tag),
                np.average(np.abs(weight)))
    except AttributeError:
        pass


class DQN(nn.Module):

    def __init__(self, observation_size, action_size):
        super(DQN, self).__init__()

        self.is_dense = len(observation_size) == 1

        if self.is_dense:
            n = 128
            self.input = nn.Linear(observation_size[0], n)
            self.feature_size = n
        else:
            self.conv1 = nn.Conv2d(
                    observation_size[0],
                    32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.feature_size = self.conv3(self.conv2(self.conv1(
                torch.zeros(1, *observation_size)))).view(1, -1).size(1)

    def forward(self, x):
        if self.is_dense:
            x = F.relu(self.input(x))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
        return x


class DQNDuelingDense(DQN):

    def __init__(
            self,
            observation_size,
            action_size,
            noisy,
            hidden_units):
        super(DQNDuelingDense, self).__init__(observation_size, action_size)

        linear = linearClass(noisy)

        self.value_fc1 = linear(self.feature_size, hidden_units)
        self.value_fc2 = linear(hidden_units, 1)
        self.advantage_fc1 = linear(self.feature_size, hidden_units)
        self.advantage_fc2 = linear(hidden_units, action_size)

    def forward(self, x):
        x = super(DQNDuelingDense, self).forward(x)

        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)

        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)

        q = value.expand_as(advantage) + (advantage -
                advantage.mean(dim=1, keepdim=True).expand_as(advantage))

        return q

    def log_scalars(self, log_fn):
        _log_noisy(log_fn, 'value_fc1', self.value_fc1)
        _log_noisy(log_fn, 'value_fc2', self.value_fc2)
        _log_noisy(log_fn, 'advantage_fc1', self.advantage_fc1)
        _log_noisy(log_fn, 'advantage_fc2', self.advantage_fc2)


class DQNDense(DQN):

    def __init__(
            self,
            observation_size,
            action_size,
            noisy,
            hidden_units):
        super(DQNDense, self).__init__(observation_size, action_size)

        linear = linearClass(noisy)
        self.fc1 = linear(self.feature_size, hidden_units)
        self.fc2 = linear(hidden_units, action_size)

    def forward(self, x):
        x = super(DQNDense, self).forward(x)
        x = F.relu(self.fc1(x))
        q = self.fc2(x)
        return q

    def log_scalars(self, log_fn):
        _log_noisy(log_fn, 'fc1', self.fc1)
        _log_noisy(log_fn, 'fc2', self.fc2)
