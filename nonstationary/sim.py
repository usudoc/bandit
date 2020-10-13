import numpy as np
import matplotlib.pyplot as plt


class BanditSimulator():
    def __init__(self, n_sims=100, n_steps=10000, n_arms=20, policy_list=None, bandit=None):
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.n_arms = n_arms
        self.policy_list = policy_list
        self.n_policy = len(policy_list)
        self.policy_name = []
        self.bandit = bandit

        self.regrets = np.zeros((self.n_policy, n_steps))

    def sim(self):
        for policy_idx, policy in enumerate(self.policy_list):
            print(policy.name)
            self.policy_name.append(policy.name)

            for sim in range(self.n_sims):
                self.bandit.initialize()
                policy.initialize()
                regrets = 0.0

                for step in range(self.n_steps):
                    self.bandit.update()  # 腕の報酬確率の更新
                    chosen_arm = policy.select_arm()  # 方策に従って腕を選択
                    reward, regret, success = self.bandit.pull(chosen_arm)  # 選択された腕を引いて報酬を得る
                    policy.update(chosen_arm, reward)  # 報酬によって

                    regrets += regret
                    self.regrets[policy_idx, step] += regrets

                print('{}: {}'.format(sim+1, regrets))

        self.regrets /= self.n_sims

    def plot(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        for j, policy_name in enumerate(self.policy_name):
            ax.plot(np.linspace(1, self.n_steps, num=self.n_steps),
                    self.regrets[j], label=policy_name,
                    linewidth=1.5, alpha=0.8)
        ax.set_xlabel('steps')
        ax.set_ylabel('regret')
        ax.set_ylim()
        ax.legend(loc='upper left', fontsize=18)
        ax.grid(axis='y')

        plt.show()