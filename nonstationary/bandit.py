import numpy as np
import random


class Bandit():
    def __init__(self, n_arms, is_stationary=True, is_grad=False, update_steps=2000):
        self.n_arms = n_arms  # 腕の数
        self.is_stationary = is_stationary  # 定常状態にするかどうか
        self.is_grad = is_grad  # 報酬確率を徐々に変化させるかどうか
        self.update_steps = update_steps  # 報酬確率を一新するstep数

        self.prob_list = np.array([0.55, 0.50, 0.45, 0.40])
        self.prob = np.array([0.55, 0.50, 0.45, 0.40])
        self.steps = 0

    # シミュレーション毎に初期化
    def initialize(self):
        self.prob = np.array([0.55, 0.50, 0.45, 0.40])
        self.steps = 0

    # 腕を引いて報酬を返す
    def pull(self, chosen_arm):
        reward = np.random.binomial(1, self.prob[chosen_arm])
        regret = np.max(self.prob) - self.prob[chosen_arm]
        best_arm = np.argmax(self.prob)
        success = 1 if chosen_arm == best_arm else 0

        return reward, regret, success

    # パラメータの更新
    def update(self):
        self.steps += 1

        # 定常状態
        if self.is_stationary:
            pass
        # 非定常状態
        else:
            # 徐々に値を変動させる
            if self.is_grad:
                self.prob = np.array([max(min(value + np.random.normal(loc=0.0, scale=0.01), 1.0), 0.0) for _, value in enumerate(self.prob)])
            # あるステップ数繰り返したら真の報酬確率を変更する
            else:
                if self.steps % self.update_steps == 0:
                    np.random.shuffle(self.prob_list)
                    self.prob = self.prob_list
