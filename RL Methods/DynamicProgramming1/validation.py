#pip install gymnasium
#pip install gymnasium\[toy-text\]

import gymnasium as gym

# actions: 0=left, 1=down, 2=right, 3=up
# rewards: Reach goal=+1, Reach hole=0, Reach frozen=0
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

