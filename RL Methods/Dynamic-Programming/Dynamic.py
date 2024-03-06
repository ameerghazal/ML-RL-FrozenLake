# Begin by importing gymnasium.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Run function, in which we pass in the number of episodes.
def run(episodes, render=False):
    # actions: 0=left, 1=down, 2=right, 3=up
    # rewards: Reach goal=+1, Reach hole=0, Reach frozen=0

    # Create the map and store.
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human" if render else None)

    # Initializes an array 16 with all 16 states.
    V = np.zeros(env.observation_space.n)

    # gamma or discount factor
    gamma = 0.9

    # Small threshold theta for determining the accuracy.
    theta = 1e-10

    # Outer loop, goes until stopped.
    while True:

      # Reinitalize the difference to 0.
      delta = 0 

      # Loop for each state in the list of states.
      for state in range(env.observation_space.n):
         
        # Setting the local value to the V(s) pair. For example, if 12 is state, then val = V(12)
        val = V[state] 

        # For each state, the V(s) is updated to the max across all possible actions. Hence, we loop for each action and use the max function. We check the max by summing the product of the transition probability, reward, and discounted value of the next-state, which we grab these values from the enviroment given our current state and current action.          
        V[state] = max(sum([probability * (reward + gamma * V[nextState]) for probability, nextState, reward, _ in env.unwrapped.P[state][action]]) for action in range(env.action_space.n))

        # Calculate the difference for the next iteration.
        delta = max(delta, abs(val - V[state]))

      # Conditional to check if we are done.
      if (delta < theta): break

    
    # Output a deterministic policy
    policy = np.zeros(env.observation_space.n, dtype=int) # Initlaize an array of size 16, with a type integer for the policy of each state.
    for state in range(env.observation_space.n): # From there, we loop for each state again.
       
       # The policy, given the state, chooses the action index that maximizes the sum of the products of the probailites, rewards, and the discounted next state. This is done for each possible next state resulting from taking different actions in the current state. The np.argmax function is then used to find the index corresponding to the action that yields the highest sum.
       policy[state] = np.argmax([sum([probability * (reward + gamma * V[nextState]) for probability, nextState, reward, _ in env.unwrapped.P[state][action]]) for action in range(env.action_space.n)]) # 
    
    print(policy)
  
# Used to run the Dynamic method.
if __name__ == '__main__':
    run(1)
