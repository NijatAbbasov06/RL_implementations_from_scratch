import gymnasium as gym
import numpy as np
import argparse


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, initial_size, grow_factor = 2):
        
        # self.obs_buf = np.zeros((initial_size, obs_dim), dtype= np.int32)
        # self.act_buf = np.zeros((initial_size, act_dim), dtype= np.int32)
        self.obs_buf = np.zeros(initial_size, dtype= np.int32)
        self.act_buf = np.zeros(initial_size, dtype= np.int32)
        self.rew_buf = np.zeros(initial_size, dtype= np.int32)
        # self.obs2_buf = np.zeros((initial_size, obs_dim), dtype= np.int32)
        self.obs2_buf = np.zeros(initial_size, dtype= np.int32)
        self.done_buf = np.zeros(initial_size, dtype= np.int32)
        self.size = initial_size
        self.ptr = 0


    def __call__(self):
        print("=" * 50)
        print("BUFFER CONTENTS")
        print("=" * 50)
        print(f"Size: {self.size}, Pointer: {self.ptr}")
        print(f"obs_buf shape: {self.obs_buf.shape}")
        print(f"act_buf shape: {self.act_buf.shape}")
        print(f"rew_buf shape: {self.rew_buf.shape}")
        print(f"obs2_buf shape: {self.obs2_buf.shape}")
        print(f"done_buf shape: {self.done_buf.shape}")
        return self

        
    def add_experience(self, o, a, r, o2, done):
        if (self.ptr >= self.size):
            self.resize()
        self.obs_buf[self.ptr] = o
        # print(o)
        
        self.act_buf[self.ptr] = a
        # print("printing r")
        # print(r)
        self.rew_buf[self.ptr] = r
        self.obs2_buf[self.ptr] = o2
        self.done_buf[self.ptr] = done
        self.ptr += 1
       
    def resize(self):
    
        # self.obs_buf = np.concatenate([self.obs_buf, 
        #             np.zeros((self.size, self.obs_buf.shape[1]), dtype= np.int32)], dtype= np.int32)
        # self.act_buf = np.concatenate([self.act_buf, 
        #             np.zeros((self.size, self.act_buf.shape[1]), dtype= np.int32)], dtype= np.int32)
        self.obs_buf = np.concatenate([self.obs_buf, 
                    np.zeros(self.size, dtype= np.int32)], dtype= np.int32)
        self.act_buf = np.concatenate([self.act_buf, 
                    np.zeros(self.size, dtype= np.int32)], dtype= np.int32)


        self.rew_buf = np.concatenate([self.rew_buf, 
                    np.zeros(self.size,dtype= np.int32)], dtype= np.int32)
        # self.obs2_buf = np.concatenate([self.obs2_buf, 
        #             np.zeros((self.size, self.obs2_buf.shape[1]), dtype= np.int32)], dtype= np.int32)
        
        self.obs2_buf = np.concatenate([self.obs2_buf, 
                    np.zeros(self.size, dtype= np.int32)], dtype= np.int32)


        self.done_buf = np.concatenate([self.done_buf, 
                    np.zeros(self.size, dtype= np.int32)], dtype= np.int32)
        self.size = self.size * 2

    # def test_buffer(self):
    #     """Test the actual ReplayBuffer class"""
    #     print("Testing ReplayBuffer...")
        
    #     # Test 1: Basic functionality
    #     print("\n1. Testing basic add_experience...")
    #     buffer = ReplayBuffer(obs_dim=4, act_dim=2, initial_size=5)
        
    #     # Add some experiences
    #     for i in range(3):
    #         obs = np.array([i, i+1, i+2, i+3], dtype=np.int32)
    #         action = np.array([i % 2], dtype=np.int32)
    #         buffer.add_experience(obs, action, r=int(i), o2=obs+1, done=(i==2))
    #         print(f"Added experience {i}, ptr: {buffer.ptr}")
        
    #     buffer()  # Print buffer contents
        
    #     # Test 2: Test resize
    #     print("\n2. Testing resize...")
    #     for i in range(3, 10):
    #         obs = np.array([i, i+1, i+2, i+3], dtype=np.int32)
    #         action = np.array([i % 2], dtype=np.int32)
    #         buffer.add_experience(obs, action, r=int(i), o2=obs+1, done=(i==9))
        
    #     buffer()  # Print buffer contents after resize
    #     print(f"Buffer size after resize: {buffer.size}")
        
    #     # Test 3: Verify data integrity
    #     print("\n3. Verifying data integrity...")
    #     print(f"First obs: {buffer.obs_buf[0]}")
    #     print(f"Last obs: {buffer.obs_buf[buffer.ptr-1]}")
    #     print(f"Total experiences: {buffer.ptr}")
        
    #     print("✓ Buffer test passed!")


    

class MonteCarlo:
    def __init__ (self, env, args, step_size, gamma, epsilon):
        self.epsilon = epsilon
        self.step_size = step_size
        self.gamma = gamma
        self.env = env
        
        
        self.obs_dim = self.env.observation_space.shape[0] if self.env.observation_space.shape else 1
        self.act_dim = self.env.action_space.shape[0] if self.env.action_space.shape else 1
        self.ReplayBuffer = ReplayBuffer(
                            obs_dim= self.obs_dim,
                            act_dim= self.act_dim,
                            initial_size= 10)
        if args.test_buffer:
            self.ReplayBuffer.test_buffer()
        
        #aliases
        self.action_value_array = 20 * np.ones((self.env.observation_space.n, self.env.action_space.n))
        
        self.value_array = np.zeros(self.env.observation_space.n)
        # print(self.action_value_array.shape)
        
        self.action_array = np.arange(self.env.action_space.n)
        self.policy_array = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        # self.return_array = np.zeros(self.env.observation_space.n, dtype= np.float32)
        

        
        delta_change = 200
        for s in range(self.policy_array.shape[0]):
            for a in range(self.policy_array.shape[1]):
                self.policy_array[s][a] = 1/ self.policy_array.shape[1]
        while delta_change>0:
            self.o,  _ = self.env.reset()
            
            done = False
            step_num_exceed = False
            
            self.num_step = 0
            while not done:
                if np.random.random() < 1 - self.epsilon:
                    a = np.argmax(self.policy_array[self.o])
                else:
                    a = np.random.choice(self.action_array)
                
                # print(self.o)
                # print(self.policy_array[self.o])
                # best_action_probability = np.max(self.policy_array[self.o])


                # # a = np.argmax(self.policy_array[self.o])
                # # print(best_action_probability)
                # best_actions = np.where(self.policy_array[self.o] == best_action_probability)[0]
                # # print(best_actions)
                # a = np.random.choice(best_actions)
                
                o2, r, done, truncated, _ = self.env.step(a)
                self.num_step +=1
                if o2 == self.o:
                    self.action_value_array[self.o, a] = 0
                
                # if o2 == self.o:
                #     done = True
                #     escape_box = True
                # print(o2)
                self.ReplayBuffer.add_experience(self.o, a, int(r), o2, done)

                self.o = o2
            self.ReplayBuffer.add_experience(self.o, a, int(r), o2, done)

            # self.ReplayBuffer()
            delta_change = self.policy_evaluation()
      
            self.policy_improvement()

    def policy_evaluation(self):
        self.G = 0
        # self.ReplayBuffer.obs_buf[self.ReplayBuffer.ptr - 1]) = self.o
        total_timestep = self.ReplayBuffer.ptr
        # print(self.ReplayBuffer.obs_buf[self.ReplayBuffer.ptr - 1])
        # print(self.o)
        delta_change = 0
        
        
        self.action_value_array[self.o] = np.zeros(self.env.action_space.n)
        for k in range(2, total_timestep + 1):
            
            

            update_index = total_timestep - k
            update_o = self.ReplayBuffer.obs_buf[update_index]
            update_a = self.ReplayBuffer.act_buf[update_index]
            update_r = self.ReplayBuffer.rew_buf[update_index]
            # update_o2 = self.ReplayBuffer.obs2_buf[update_index]

            self.G = update_r + self.gamma * self.G
            
            

            # print(bootstrap_state)
            # print("update_index")
            # print(update_index)
            
            
            mc_error = self.G - self.action_value_array[update_o][update_a]
            
            self.action_value_array[update_o][update_a] = (self.action_value_array[update_o][update_a]
                                            + self.step_size * mc_error)
            # print(self.action_value_array)
            delta_change = max(delta_change, abs(mc_error))

            self.get_value_function()

            self.print_value_grid()
            
        return delta_change


    def policy_improvement(self):
        for s in range(self.env.observation_space.n):
            # action_values = [self.action_value_array[s][a] for a in self.action_array]
            action_values = self.action_value_array[s]
            # print(self.action_value_array[s])
            print(action_values)

            best_action_value = np.max(action_values)
            
            best_actions = np.where(action_values == best_action_value)[0]
            
            best_action = np.random.choice(best_actions)
            # best_action = np.argmax(action_values)
            print(best_action)
            for a in self.action_array:
                if a == best_action:
                    self.policy_array[s][a] = 1
                else:
                    # print("yeah")
                    self.policy_array[s][a] = 0
        # print(self.policy_array)
        # print(self.policy_array)

    def get_value_function(self):
        
        for state in range(self.env.observation_space.n):
            self.value_array[state] = np.max(self.action_value_array[state])
        
            

        
        # print(self.o)
        
    # def print_grid_q_values(self, q_values, grid_size=(8,8)):
    #     print("\nQ-TABLE GRID:")
    #     actions = ["up", "right", "down", "left"]  # Visual arrows
        

        
    #     for row in range(grid_size[0]):
    #         for col in range(grid_size[1]):
    #             best_action = np.argmax(q_values[row])
               
    #             best_value = q_values[row][best_action]
    #             print(f"({row},{col}){actions[best_action]}", end="  ")
                
    #         print()  # New line after each row 
    def print_value_grid(self):
        """Print value function as a clean grid"""
        size = int(np.sqrt(self.env.observation_space.n))
        grid = self.value_array.reshape((size, size))
        
        print("\n" + "="*40)
        print("VALUE FUNCTION")
        print("="*40)
        for row in grid:
            print(" ".join(f"{val:6.3f}" for val in row))

    def choose_action(state, Q, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(env.action_space.n)  # Explore
        else:
            return np.argmax(Q[state])  # Exploit

    
# Add this to your ReplayBuffer class





if __name__ == "__main__" :

    parser = argparse.ArgumentParser(description="RL training with "
                                     "Testing Options")
    parser.add_argument( "--test-buffer", 
                        action = "store_true", 
                        help = "Test if the buffer is setup properly")
    
    parser.add_argument("--test-MC", 
                        action= "store_true",
                        help = "Test if the buffer is setup properly")
    
    
    args = parser.parse_args()

    
    env = gym.make(
        'FrozenLake-v1',
        desc= None,
        map_name = "4x4",
        is_slippery=False,
        render_mode = "human"
        )

    algorithm = MonteCarlo(env, args, step_size = 0.01, gamma = 0.99, epsilon = 0.2)

    