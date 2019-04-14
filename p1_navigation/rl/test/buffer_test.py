from unittest import TestCase

import numpy as np
import torch

from rl import ReplayBuffer

class TestReplayBuffer(TestCase):

    def test_push(self):
        b = ReplayBuffer(2)
        b.push(np.array([1]), 2, 3, np.array([4]))
        b.push(np.array([2]), 3, 4, np.array([5]))
        b.push(np.array([3]), 4, 5, np.array([6]))
        np.random.seed(42)

        states, actions, rewards, next_states = b.sample(2)

        self.assertTrue(torch.equal(
            torch.tensor([[2], [3]]),
            states))
        self.assertTrue(torch.equal(
            torch.tensor([[3], [4]]),
            actions))
        self.assertTrue(torch.equal(
            torch.tensor([[4], [5]]),
            rewards))
        self.assertTrue(torch.equal(
            torch.tensor([[5], [6]]),
            next_states))

    def test_len(self):
        b = ReplayBuffer(2)

        self.assertEqual(len(b), 0)

        b.push(1, 2, 3, 4)
        self.assertEqual(len(b), 1)

        b.push(2, 3, 4, 5)
        self.assertEqual(len(b), 2)

        b.push(3, 4, 5, 6)
        self.assertEqual(len(b), 2)
