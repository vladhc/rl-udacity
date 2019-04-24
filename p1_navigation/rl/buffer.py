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
        return self._sample(indexes)

    def _sample(self, indexes):
        states = [self._buffer[idx].state for idx in indexes]
        actions = [self._buffer[idx].action for idx in indexes]
        rewards = [self._buffer[idx].reward for idx in indexes]
        next_states = [self._buffer[idx].next_state for idx in indexes]
        return states, actions, rewards, next_states, indexes

    def __len__(self):
        return len(self._buffer)


EPSILON = 0.0001
BIG_NUMBER = 10000000.0


class PriorityReplayBuffer(ReplayBuffer):

    def __init__(self, capacity, beta=1.0):
        """ beta: Prioritization importance sampling """
        ReplayBuffer.__init__(self, capacity)
        self._priorities = []
        self._beta = beta

    def set_beta(self, beta):
        self._beta = beta

    def update_priorities(self, indexes, priorities):
        for i in range(len(indexes)):
            idx = indexes[i]
            self._priorities[idx] = priorities[i]

    def importance_sampling_weights(self, indexes):
        # p = self._probabilities()
        p = self._p
        p = [p[idx] for idx in indexes]
        w = [(1 / prob) ** self._beta for prob in p]
        max_w = np.max(w)
        w = [x / max_w for x in w]
        return np.asarray(w)

    def _probabilities(self):
        priorities = [(pr + EPSILON) for pr in self._priorities]
        s = np.sum(priorities)
        assert s > 0
        p = [(priority / s) for priority in priorities]
        return p

    def sample(self, batch_size):
        p = self._probabilities()
        self._p = p
        indexes = np.random.choice(len(self._buffer), size=batch_size, p=p)
        return self._sample(indexes)

    def push(self, *args):
        t = Transition(*args)

        while len(self) >= self._capacity:
            idx = np.argmin(self._priorities)
            self._buffer.pop(idx)
            self._priorities.pop(idx)

        priority = np.max(self._priorities) if len(self._priorities) != 0 \
                                            else BIG_NUMBER
        self._buffer.append(t)
        self._priorities.append(priority)
