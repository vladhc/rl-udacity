import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Though the code was shamelessly taken from here:
# https://github.com/Shmuma/ptan/blob/master/samples/rainbow/04_dqn_noisy_net.py
# the source paper was carefully read through


class NoisyLinear(nn.Linear):

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(
                torch.Tensor(out_features, in_features).fill_(sigma_init))
        self.register_buffer(
                "epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(
                    torch.Tensor(out_features).fill_(sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        nn.init.uniform(self.weight, -std, std)
        nn.init.uniform(self.bias, -std, std)

    def forward(self, input):
        torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
        bias = self.bias
        if bias is not None:
            torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
            bias = bias + self.sigma_bias * Variable(self.epsilon_bias)
        return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), bias)
