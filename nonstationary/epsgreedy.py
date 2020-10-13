import numpy as np

from base_policy import BasePolicy


class EpsGreedy(BasePolicy):
    def __init__(self, n_arms, eps=0.1):
        super().__init__(n_arms)
        self.name = 'EpsGreedy ε={}'.format(eps)

        self._eps = eps
        self.eps = eps

    def initialize(self):
        super().initialize()
        self.eps = self._eps

    def _greedy(self, q):
        max_idx = np.where(q == np.max(q))
        return np.random.choice(max_idx[0])

    def select_arm(self):
        if True in (self.counts < self.warmup):
            return np.argmax(np.array(self.counts < self.warmup))
        # 一様ランダム選択
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.n_arms)
        # greedy選択
        else:
            return self._greedy(self.q)

    def update(self, chosen_arm, reward):
        super().update(chosen_arm, reward)
        self.q[chosen_arm] += 1 / self.counts[chosen_arm] * (reward - self.q[chosen_arm])
        self.eps *= 0.999
