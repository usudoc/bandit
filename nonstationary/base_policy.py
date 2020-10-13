import numpy as np


class BasePolicy():
    def __init__(self, n_arms, warmup=1):
        self.n_arms = n_arms
        self.warmup = warmup
        self.q = np.zeros(n_arms)
        self.steps = 0
        self.counts = np.zeros(n_arms)
        self.name = None

    def initialize(self):
        self.q = np.zeros(self.n_arms)
        self.steps = 0
        self.counts = np.zeros(self.n_arms)

    def select_arm(self):
        pass

    def update(self, chosen_arm, reward):
        self.steps += 1
        self.counts[chosen_arm] += 1
