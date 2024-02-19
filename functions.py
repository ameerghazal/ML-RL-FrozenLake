# Import gymnasium
import gymnasium as gym

# Create the map using the make() function.
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

# reset(): resets the map.
env.reset()

# render(): displays the map / environment.
env.render()

# returns the states (obs. space).
env.observation_space

# actions: left = 0, down = 1, right = 2, up = 3
env.action_space

# Generates a random action with the sample() function
randomAction = env.action_space.sample()

# Step(action) returns a return value:
returnVal = env.step(randomAction)
# format of returnValue is (observation,reward, terminated, truncated, info)
# observation (object)  - observed state
# reward (float)        - reward that is the result of taking the action
# terminated (bool)     - is it a terminal state
# truncated (bool)      - it is not important in our case
# info (dictionary)     - in our case transition probability

# We can also specify a deterministic step in {0,1,2,3}
env.step(1)

# p() will return our transition probabilites.
# Note, p(s' | s,a) is the probability of going to the next state s', starting from state s and by applying the action a.
# In V-1.0, we must use the unwrapped attribute followed by P[state][action]
# Output is a list having the following entries
# (transition probability, next state, reward, Is terminal state?)
print(env.unwrapped.P[9][0])