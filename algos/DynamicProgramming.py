import numpy as np
import gymnasium as gym 
import time



class DynamicProgramming:
    def __init__(self, env, gamma = 0.99):

        self.gamma = gamma
        self.env = env
       
        

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        assert self.env.P != None, "state_transition probabilities unknown"
        #transition_matrix_alias
        self.P = self.env.P
        self.reset()
        delta_change = 100
        while delta_change > 0.0000001:
            delta_change = self.policy_evaluation()
            # print(delta_change)
        self.policy_improvement()
        print(self.value_array)

    def __call__(self):
        done = False
        while not done:
            s, r, done, _, _ = self.env.step(int(np.argmax(self.policy_array[self.current_state])))
            
            print("self.policy_array[self.current_state]:     " + str(self.policy_array[self.current_state]))

            self.current_state = s
            print("self.current_state:   " + str(self.current_state))
            time.sleep(1)
        self.print_value_grid()
        return self.policy_array
    

        
      
    def policy_evaluation(self):

        delta_change = 0
        for s in range(self.observation_space.n):
        

            action_values = [
                sum(self.compute_bootstrapped_value(transition) 
                for transition in self.P[s][a])
                for a in self.action_array
            ]

            best_action = np.argmax(action_values)
            
            
            best_value = action_values[best_action]
            # print(best_value)
            delta_change = max(delta_change, abs(best_value - self.value_array[s]) )


            self.value_array[s] = best_value

        return delta_change
            



        pass
    def policy_improvement(self):
        for s in range(self.observation_space.n):
            action_values = [
                sum(self.compute_bootstrapped_value(transition)
                for transition in self.P[s][a]) 
                for a in self.action_array
            ]
            if s == 8:
                print(s)
                print(action_values)



            best_action = np.argmax(action_values)
            print(best_action)
            for a in self.action_array:
                if a == best_action:
                    self.policy_array[s][a] = 1
                else:
                    self.policy_array[s][a] = 0
        
        
        pass

    def reset(self):
        self.current_state, _= self.env.reset()
      

        self.value_array_size = self.observation_space.n
        self.action_array_size = self.action_space.n
        self.value_array = np.zeros(self.value_array_size)
        self.action_array = np.arange(self.action_space.n)
        
        
        self.policy_array = np.zeros((self.value_array_size, self.action_array_size))
        #Policy Array initialization
        for i in range(self.value_array_size):
            for j in range(self.action_array_size):
                self.policy_array[i][j] = 1/self.action_array_size

    
    def compute_bootstrapped_value(self, transition):
        # print(transition[0] * (transition[2] + self.gamma * self.value_array[transition[1]]))

        return transition[0] * (transition[2] + self.gamma * self.value_array[transition[1]])
    
    def print_value_grid(self):
        """Print values in a grid format"""
        size = int(np.sqrt(self.observation_space.n))
        grid = self.value_array.reshape((size, size))
        print("Value Grid:")
        for row in grid:
            print(" ".join(f"{val:.3f}" for val in row))


        
if __name__ == "__main__":
    env = gym.make(
    'FrozenLake-v1',
    desc= None,
    map_name = "4x4",
    is_slippery=False,
    render_mode = "human"
    )


    print("running the algo")
    
    algorithm = DynamicProgramming(env)
    policy_array = algorithm()
    # print("policy array")
    # print(policy_array)
    # print(env.P)


    success_count = 0
    for eisode in range(1000):
        state = env.reset()[0]
        done = False
        while not done:
            action = np.argmax(algorithm.policy_array[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward == 1:
            success_count += 1

        print(f"Success rate: {success_count}/1000 = {success_count/10}%")
    env.close()

