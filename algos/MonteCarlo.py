import gymnasium as gym
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, initial_size, grow_factor = 2):
        self.obs_buf = np.zeros((initial_size, obs_dim), dtype= np.float32)
        self.act_buf = np.zeros((initial_size, act_dim), dtype= np.float32)
        self.rew_buf = np.zeros(initial_size, dtype= np.float32)
        self.obs2_buf = np.zeros((initial_size, obs_dim), dtype= np.float32)
        self.done_buf = np.zerps(initial_size, dtype= np.float32)
        self.size = initial_size
        self.ptr = 0
    def add_experience(self, o, a, r, o2, done):
        if (self.ptr > self.size):
            self.resize()
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.obs2_buf[self.ptr] = o2
        self.done_buf[self.ptr] = done
        self.ptr += 1
       
    def resize(self):
        add_buf = np.zeros((self.size), dtype= np.float32)
        self.obs_buf = np.concatenate([self.obs_buf, add_buf], dtype= np.float32)
        self.act_buf = np.concatenate([self.act_buf, add_buf], dtype= np.float32)
        self.rew_buf = np.concatenate([self.rew_buf, add_buf], dtype= np.float32)
        self.obs2_buf = np.concatenate([self.obs2_buf, add_buf], dtype= np.float32)
        self.size = self.size * 2

    

class MonteCarlo:
    def __init__ (self, env):
        self.env = env
        #aliases
    
        self.value_array = np.zeros(self.env.observation_space.n)
        self.action_array = np.arange(self.env.action_space.n)
        self.policy_array = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        
        delta_change = 200
        while delta_change>0.001:
            self.reset()
            a = np.argmax(self.policy_array[s])
            o2, r, done, truncated, _ = self.env.step(a)
            self.policy_evaluation(self.current_state, a, r, o2, done, truncated)
            self.policy_improvement()
    def policy_evaluation(self, o, a, r, o2, done, truncated):
        


        pass
    def policy_improvement(self):
        pass

    def reset(self):
        self.o,  _ = self.env.reset()
        for s in range(self.policy_array.shape[0]):
            for a in range(self.policy_array.shape[1]):
                self.policy_array[s][a] = 1/ self.policy_array.shape[1]
        



        
        





if __name__ == "__main__" :
    env = gym.make(
    'FrozenLake-v1',
    desc= None,
    map_name = "8x8",
    is_slippery=True,
    render_mode = "human"
    )

    algorithm = MonteCarlo(env)