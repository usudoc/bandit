import numpy as np

from base_policy import BasePolicy


class RS(BasePolicy):
    def __init__(self, n_arms, aleph=0.8):
        super().__init__(n_arms)
        self.name = 'RS ℵ={}'.format(aleph)

        self.aleph = aleph

    def initialize(self):
        super().initialize()

    def _greedy(self, q):
        max_idx = np.where(q == np.max(q))
        return np.random.choice(max_idx[0])

    def select_arm(self):
        if True in (self.counts < self.warmup):
            return np.argmax(np.array(self.counts < self.warmup))
        rs = self.counts * (self.q - self.aleph)

        return self._greedy(rs)

    def update(self, chosen_arm, reward):
        super().update(chosen_arm, reward)
        self.q[chosen_arm] += 1 / self.counts[chosen_arm] * (reward - self.q[chosen_arm])
