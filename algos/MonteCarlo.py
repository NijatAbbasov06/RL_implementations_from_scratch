import gymnasium as gym
import numpy as np


class MonteCarlo:
    def __init__ (self, env):
        self.env = env
    def policy_evaluation(self):
        pass
    def policy_improvement(self):
        pass





if __name__ == "__main__" :
    gym.make('Blackjack-v1')