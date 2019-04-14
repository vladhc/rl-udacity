import torch.nn.functional as F
import torch.nn as nn

HIDDEN_UNITS = 64


class DQNDense(nn.Module):

    def __init__(self, observation_size, action_size):
        super(DQNDense, self).__init__()
        self.fc1 = nn.Linear(observation_size, HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.fc3 = nn.Linear(HIDDEN_UNITS, action_size)

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x