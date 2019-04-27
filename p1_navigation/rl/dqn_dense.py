import torch.nn.functional as F
import torch.nn as nn
from rl import NoisyLinear

HIDDEN_UNITS = 128


def linearClass(noisy):
    if noisy:
        return NoisyLinear
    else:
        return nn.Linear


class DQNDuelingDense(nn.Module):

    def __init__(self, observation_size, action_size, noisy):
        super(DQNDuelingDense, self).__init__()
        assert len(observation_size) == 1

        self.input = nn.Linear(observation_size[0], HIDDEN_UNITS)

        linear = linearClass(noisy)
        n = int(HIDDEN_UNITS/2)

        self.value_fc1 = linear(HIDDEN_UNITS, n)
        self.value_fc2 = linear(n, 1)
        self.advantage_fc1 = linear(HIDDEN_UNITS, n)
        self.advantage_fc2 = linear(n, action_size)

    def forward(self, x):
        x = x.float()

        x = F.leaky_relu(self.input(x))

        value = F.leaky_relu(self.value_fc1(x))
        value = self.value_fc2(value)

        advantage = F.leaky_relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)

        q = value.expand_as(advantage) + (advantage -
                advantage.mean(dim=1, keepdim=True).expand_as(advantage))

        return q


class DQNDense(nn.Module):

    def __init__(self, observation_size, action_size, noisy):
        super(DQNDense, self).__init__()
        assert len(observation_size) == 1

        linear = linearClass(noisy)

        self.input = nn.Linear(observation_size[0], HIDDEN_UNITS)

        self.fc1 = linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.fc2 = linear(HIDDEN_UNITS, action_size)

    def forward(self, x):
        x = x.float()

        x = self.input(x)
        x = F.leaky_relu(x)

        x = F.leaky_relu(self.fc1(x))
        q = self.fc2(x)

        return q
