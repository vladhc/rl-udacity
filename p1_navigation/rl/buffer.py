import numpy as np
from collections import namedtuple

Transition = namedtuple(
        'Transition',
        ('state', 'action', 'reward', 'next_state', 'trajectory_id', 'step'))


class ReplayBuffer:

    def __init__(self, capacity):
        self._capacity = capacity
        self._buffer = []
        self._trajectory_id = 0
        self._step = 0

    def push(self, state, action, reward, next_state, done):
        t = self._transition(state, action, reward, next_state, done)
        while len(self) >= self._capacity:
            self._buffer.pop(0)
        self._buffer.append(t)

    def _transition(self, state, action, reward, next_state, done):
        t = Transition(
                state,
                action,
                reward,
                next_state,
                self._trajectory_id,
                self._step)
        self._step += 1
        if done:
            self._trajectory_id += 1
            self._step = 0
        return t

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
P0 = 1.0


class PriorityReplayBuffer(ReplayBuffer):

    def __init__(self, capacity, alpha=0.5, beta=1.0):
        """
        alpha: prioritization exponent. How much prioritization is used.
               alpha = 0 → uniform
               alpha = 1 → prioritirized
        beta: Prioritization importance sampling
        """
        ReplayBuffer.__init__(self, capacity)
        self._idx = 0
        self._priorities = np.zeros(capacity, dtype=float)
        self._beta = beta
        self._alpha = alpha

    def set_beta(self, beta):
        self._beta = beta

    def update_priorities(self, indexes, priorities):
        priorities = priorities + EPSILON
        self._priorities[indexes] = priorities

    def importance_sampling_weights(self, indexes):
        p = self._probabilities()
        p = p[indexes]
        n = len(self)
        w = np.power(n * p, -self._beta)
        w = w / np.max(w)
        return w

    def _probabilities(self):
        p = self._priorities[:self._idx]
        p = np.power(p, self._alpha)
        return p / np.sum(p)

    def sample(self, batch_size):
        p = self._probabilities()
        s = min(len(self._buffer), batch_size)
        indexes = np.random.choice(self._idx, size=s, p=p, replace=True)
        return self._sample(indexes)

    def push(self, state, action, reward, next_state, done):
        t = self._transition(state, action, reward, next_state, done)

        if len(self) == self._capacity:
            idx = np.argmin(self._priorities)
            self._buffer[idx] = t
            self._priorities[idx] = np.max(self._priorities)
        else:
            idx = self._idx
            priority = np.max(self._priorities) if len(self) != 0 else P0
            self._buffer.append(t)
            self._priorities[idx] = priority
            self._idx += 1
