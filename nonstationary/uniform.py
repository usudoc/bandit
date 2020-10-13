import numpy as np

from base_policy import BasePolicy


class Uniform(BasePolicy):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.name = 'Uniform'

    def select_arm(self):
        return np.random.randint(0, self.n_arms, 1)

    def update(self, chosen_arm, reward):
        pass
