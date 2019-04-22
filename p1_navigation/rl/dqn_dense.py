import torch.nn.functional as F
import torch.nn as nn
import torch
from rl import NoisyLinear

HIDDEN_UNITS = 64


def linearClass(noisy):
    if noisy:
        return NoisyLinear
    else:
        return nn.Linear


class DQNDuelingDense(nn.Module):

    def __init__(self, observation_size, action_size, noisy):
        super(DQNDuelingDense, self).__init__()
        self._action_size = action_size

        self.fc1 = nn.Linear(observation_size, HIDDEN_UNITS)

        linear = linearClass(noisy)
        n = int(HIDDEN_UNITS/2)

        self.value_fc1 = linear(HIDDEN_UNITS, n)
        self.value_fc2 = linear(n, 1)
        self.advantage_fc1 = linear(HIDDEN_UNITS, n)
        self.advantage_fc2 = linear(n, action_size)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))

        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)

        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        avg = advantage.mean(dim=1)
        avg = torch.unsqueeze(avg, dim=1)
        avg = avg.expand(-1, self._action_size)
        advantage = advantage - avg

        q = value + advantage

        return q

class DQNDense(nn.Module):

    def __init__(self, observation_size, action_size, noisy):
        super(DQNDense, self).__init__()

        linear = linearClass(noisy)

        self.fc1 = linear(observation_size, HIDDEN_UNITS)
        self.fc2 = linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.fc3 = linear(HIDDEN_UNITS, action_size)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        q = self.fc3(x)

        return q
