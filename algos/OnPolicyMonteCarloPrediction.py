
import sys
import os

sys.path.insert(1, "../")

current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)

sys.path.insert(1, parent_dir)
import gymnasium as gym
import numpy as np
import argparse
import time
from utils.wrapper import JupyterRender

from collections import defaultdict




#This implementation includes both first_visit MonteCarlo and every_visit MonteCarlo

#Initialize Value function with shape (observation_space, 1)
#Initialize returns dictionary with shape (observation_space, [number of visit]) (we append the returns as we visit the states)
#Loop in range episode_count
#Simulate given policy and record a trajectory with tuples (state, action, reward)
#G=0
#G <- R + gamma* G
#append G to the returns[state]
#end loop
#average returns for all states

#for each observation state find the average of the returns

class MonteCarloPrediction:
    def __init__(self, env, args,  policy, step_size = 0.01, gamma = 0.99, epsilon = 0.1):
        self.env = env
        self.step_size = step_size
        self.policy = policy
        self.epsilon = epsilon
        self.gamma = gamma
        self.episode_count = args.episode_count
        self.type = args.type
        self.v_values = np.zeros([self.env.observation_space.n])
        self.returns = defaultdict(list)
        
        self.loop()
        self.average()


        pass

    def __call__(self):
        self.env.render(v=self.v_values, policy=self.policy)
        time.sleep(0.4)



        pass

    def loop(self):
        for episode in range(self.episode_count):
            self.simulate_policy()
            if self.type == "every_visit":
                self.return_sampling_every_visit()
            elif self.type == "first_visit":
                self.return_sampling_first_visit()
            
   

   

    def simulate_policy(self):
        o, _ =self.env.reset()
        self.trajectory = []
        done = False
        while not done:
            o2, r, done, _, _ = env.step(self.policy[o])
            self.trajectory.append((o, r))
            o = o2
        
    def return_sampling_first_visit(self):
        G=0
        visited = set()
        for o, r in reversed(self.trajectory):
            
            G = self.gamma * G + r
            if o not in visited:
                visited.add(o)
                self.returns[o].append(G)


    
    def return_sampling_every_visit(self):
        G = 0
        
        for o, r in reversed(self.trajectory):
           
            G = self.gamma * G +  r
            self.returns[o].append(G)


    def average(self):
        for state, G_list in self.returns.items():
            if G_list:  # Avoid division by zero
                self.v_values[state] = np.mean(G_list)
                
            

        pass
   








if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="RL training with "
                                     "Testing Options")
    parser.add_argument( "--test-buffer", 
                        action = "store_true", 
                        help = "Test if the buffer is setup properly")
    
    parser.add_argument("--test-MC", 
                        action= "store_true",
                        help = "Test if the buffer is setup properly")
    
    parser.add_argument( "--episode_count",
                        type= int,
                        default= 20,
                        help = "Episode count to train the model")
    

    parser.add_argument("--type",
                        type = str,
                        default= "every_visit",
                        help= "Specify either First visit or Every visit MonteCarlo")
    
    args = parser.parse_args()

    
    env = gym.make(
        'FrozenLake-v1',
        desc= None,
        map_name = "4x4",
        is_slippery=False,
        render_mode = "rgb_array"
        )
    
    env = JupyterRender(env)

    policy = np.array([1, 2, 1, 0, 1, 0, 1, 0, 2, 1, 1, 0, 0, 2, 2, 0], dtype=int) 
    algorithm = MonteCarloPrediction(env, args, step_size = 0.01, gamma = 0.7, epsilon = 0.1, policy= policy)
    algorithm()
    # policy_array = algorithm()

    # success_count = 0
    # for eisode in range(1000):
    #     state = env.reset()[0]
    #     done = False
    #     while not done:
    #         action = np.argmax(algorithm.policy_array[state])
    #         state, reward, terminated, truncated, _ = env.step(action)
    #         done = terminated or truncated
    #     if reward == 1:
    #         success_count += 1

    #     print(f"Success rate: {success_count}/1000 = {success_count/10}%")
    env.close()