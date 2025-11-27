import gymnasium as gym 

import time
# env = gym.make(
#     'FrozenLake-v1', 
#     desc = None, 
#     map_name = "4x4",
#     is_slippery = True, 
#     success_rate = 1.0/3.0,
#     reward_schedule = (1, 0, 0)
# )

env = gym.make(
    'FrozenLake-v1',
    desc=None,
    map_name="4x4",
    is_slippery=True,
    render_mode = "human"
)


print(env.P[1][1])



result = env.reset()
env.render()
result_step = env.step(0)
print(result)
print(result_step)



time.sleep(10)