import torch.nn.functional as F
import torch.nn as nn
import torch

HIDDEN_UNITS = 64


class DQNDense(nn.Module):

    def __init__(self, observation_size, action_size):
        super(DQNDense, self).__init__()
        self._action_size = action_size

        self.fc1 = nn.Linear(observation_size, HIDDEN_UNITS)

        n = int(HIDDEN_UNITS/2)

        self.value_fc1 = nn.Linear(HIDDEN_UNITS, n)
        self.value_fc2 = nn.Linear(n, 1)
        self.advantage_fc1 = nn.Linear(HIDDEN_UNITS, n)
        self.advantage_fc2 = nn.Linear(n, action_size)

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
