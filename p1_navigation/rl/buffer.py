import numpy as np


class ReplayBuffer:

    def __init__(self, capacity, observation_shape):
        self._capacity = capacity

        self._states = np.zeros(
                (capacity,) + observation_shape, dtype=np.float16)
        self._next_states = np.zeros(
                (capacity,) + observation_shape, dtype=np.float16)
        self._actions = np.zeros(capacity, dtype=np.uint8)
        self._rewards = np.zeros(capacity, dtype=np.float16)
        self._term = np.zeros(capacity, dtype=np.uint8)

        self.reset()

    def reset(self):
        self._cursor = 0
        self._overwrite = False

    def push(self, state, action, reward, next_state, done):
        idx = self._cursor

        self._states[idx] = state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._term[idx] = 1 if done else 0

        self._cursor += 1
        if self._cursor >= self._capacity:
            self._cursor = 0
            self._overwrite = True

        return idx

    def capacity(self):
        return self._capacity

    def sample(self, batch_size):
        s = len(self)
        batch_size = min(s, batch_size)
        indexes = np.random.choice(s, size=batch_size, replace=True)
        return self._sample(indexes)

    def _sample(self, indexes):
        states = self._states[indexes]
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        next_states = self._next_states[indexes]
        term = self._term[indexes]
        return states, actions, rewards, next_states, term, indexes

    def __len__(self):
        if self._overwrite:
            return self.capacity()
        else:
            return self._cursor


EPSILON = 0.0001
P0 = 1.0


class PriorityReplayBuffer(ReplayBuffer):

    def __init__(self, capacity, observation_shape, alpha=0.5, beta=1.0):
        """
        alpha: prioritization exponent. How much prioritization is used.
               alpha = 0 → uniform
               alpha = 1 → prioritirized
        beta: Prioritization importance sampling
        """
        ReplayBuffer.__init__(self, capacity, observation_shape)
        self._priorities = np.zeros(capacity, dtype=float)
        self._beta = beta
        self._alpha = alpha

    def set_beta(self, beta):
        self._beta = beta

    def update_priorities(self, indexes, priorities):
        priorities = np.maximum(priorities, EPSILON)
        self._priorities[indexes] = priorities

    def importance_sampling_weights(self, indexes):
        p = self._probabilities()
        p = p[indexes]
        n = len(self)
        w = np.power(n * p, -self._beta)
        w = w / np.max(w)
        return w

    def _probabilities(self):
        p = self._priorities[:len(self)]
        p = np.power(p, self._alpha)
        return p / np.sum(p)

    def sample(self, batch_size):
        indexes = np.random.choice(
                len(self),
                size=min(len(self), batch_size),
                p=self._probabilities(),
                replace=True)
        return self._sample(indexes)

    def push(self, state, action, reward, next_state, done):
        idx = super(PriorityReplayBuffer, self).push(
                state, action, reward, next_state, done)

        priority = np.max(self._priorities) if len(self) != 1 else P0
        self._priorities[idx] = priority
