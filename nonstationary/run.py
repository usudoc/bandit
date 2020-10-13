from sim import BanditSimulator
from bandit import Bandit
from uniform import Uniform
from epsgreedy import EpsGreedy
from rs import RS


if __name__ == '__main__':

    n_sims = 20
    n_steps = 10000
    n_arms = 4

    policy_list = [
        # Uniform(n_arms),
        EpsGreedy(n_arms, eps=0.5)
        , RS(n_arms, aleph=0.53)
    ]
    bandit = Bandit(n_arms, is_stationary=False, is_grad=False,
                    update_steps=2000)
    simulator = BanditSimulator(n_sims=n_sims, n_steps=n_steps, n_arms=n_arms, policy_list=policy_list, bandit=bandit)
    simulator.sim()
    simulator.plot()