import torch
import numpy as np
from collections import namedtuple

Transition = namedtuple(
        'Transition',
        ('state', 'action', 'reward', 'next_state'))


class ReplayBuffer:

    def __init__(self, capacity):
        self._capacity = capacity
        self._buffer = []

    def push(self, *args):
        t = Transition(*args)
        while len(self) >= self._capacity:
            self._buffer.pop(0)
        self._buffer.append(t)

    def capacity(self):
        return self._capacity

    def sample(self, batch_size):
        batch_size = min(len(self), batch_size)
        indexes = np.random.choice(
                len(self._buffer),
                size=batch_size,
                replace=True)
        states = [self._buffer[idx].state for idx in indexes]
        actions = [self._buffer[idx].action for idx in indexes]
        rewards = [self._buffer[idx].reward for idx in indexes]
        next_states = [self._buffer[idx].next_state for idx in indexes]
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self._buffer)
