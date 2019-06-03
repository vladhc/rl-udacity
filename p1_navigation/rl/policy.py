import torch
import numpy as np


class GreedyPolicy:

    def __init__(self):
        pass

    def get_action(self, q_values):
        best_actions = np.argmax(q_values, axis=1)
        return best_actions


class EpsilonPolicy:

    def __init__(
            self,
            policy,
            action_size,
            epsilon_start=0.5, epsilon_end=0.1, epsilon_decay=200):
        self._policy = policy
        self._step = 0
        self._action_size = action_size
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._epsilon = self._epsilon_start

    def get_action(self, q_values):
        self._epsilon = self._epsilon_end + \
            (self._epsilon_start - self._epsilon_end) * \
            np.exp(-1. * self._step / self._epsilon_decay)
        self._step += 1

        if np.random.uniform() > self._epsilon:
            return self._policy.get_action(q_values)
        else:
            return np.random.randint(
                    self._action_size,
                    size=(len(q_values,)))

    def get_epsilon(self):
        return self._epsilon
