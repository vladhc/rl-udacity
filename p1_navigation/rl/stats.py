from collections import defaultdict
import numpy as np


class Statistics(object):

    def __init__(self):
        self._dict = defaultdict(list)

    def set(self, key, value):
        k = self._dict[key]
        k.append(value)
        self._dict[key] = k

    def avg(self, key):
        return np.average(self._dict[key])

    def sum(self, key):
        return sum(self._dict[key])

    def max(self, key):
        return max(self._dict[key])
