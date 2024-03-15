# Begin by importing gymnasium.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Decay function to test for different functions.
def decayFunction(epsilon, epsilonDecay, type="linear"):
    if (type == "exponetial"): # Exponential decay.
      return epsilon * np.exp(-epsilonDecay)
    
    if (type == "inverse"): # Inverse decay.
      return epsilon / (1 + epsilonDecay)
    
    return epsilon - epsilonDecay # Last case, return linear-step decay.

# Run function, in which we pass in the number of episodes.
def doubleQ(episodes, decayParam = (1, 0.0001, "linear"), render=False):
    # actions: 0=left, 1=down, 2=right, 3=up
    # rewards: Reach goal=+1, Reach hole=0, Reach frozen=0

    # Create the map and store.
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human" if render else None)

    # Initializes an array 16 x 4 with zeroes; Q(s,a) for all state and actions = 0
    Q1 = np.zeros((env.observation_space.n, env.action_space.n))
    Q2 = np.zeros((env.observation_space.n, env.action_space.n))

    # alpha or learning rate or step-size
    alpha = 0.1

    # gamma or discount factor
    gamma = 0.9

    #100% random actions; epsilon is the value that determines exploration vs. explotation.
    epsilon = decayParam[0]
    epsilonDecay = decayParam[1] #epsilon decay rate | 1/0.0001 = 10,000
    randomNum = np.random.default_rng()

    # For graph: intialize the rewards, epsilons to 0.
    rewardsPerEpisode = np.zeros(episodes)
    avgRewardPerEp = np.zeros(episodes) # Initalize list of zeros, holding the average reward per epiosde.
    successfulEpisodes = 0  # Number of successful episodes, where the agent reaches the goal.
    stepsToReachGoal = [] # List containing the number of steps to reach the goal per episode. 

    for i in range(episodes):
        state = env.reset()[0] #states 0:15
        terminated = False #True when agent falls in hole or reaches the goal
        truncated = False  #True if steps is greater than 100
        steps = 0 # Counter for the number of steps in the episode, used for the average later.

        # This entire iteration is an episode.
        while(not terminated and not truncated):

            # Check if a random number (between 0-0.99) is less than epsilon. # If so, we take a random action (explore).
            if randomNum.random() < epsilon:
                action = env.action_space.sample()  # generate random action | actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = np.argmax(Q1[state, :] + Q2[state, :])  # Otherwise, we take a greedy action based on the data we have from Q matrix (exploit).
      
            # Take action A, observe reward and next state.
            nextState,reward,terminated,truncated,_ = env.step(action)

            if randomNum.random() < 0.5:
                # Q1(s,a) = Q1(s,a) + alpha(Reward + gamma * Q2(s', argmax(Q1(S', a)) - Q1(s,a))
                Q1[state, action] = Q1[state, action] + alpha * (reward + gamma * Q2[nextState, np.argmax(Q1[nextState, :])] - Q1[state, action])
            else:
                # Q2(s,a) = Q2(s,a) + alpha(Reward + gamma * Q1(s', argmax(Q2(S', a)) - Q2(s,a))
                Q2[state, action] = Q2[state, action] + alpha * (reward + gamma * Q1[nextState, np.argmax(Q1[nextState, :])] - Q2[state, action])

            # S = S', A = A'
            state = nextState
            steps += 1 # Increment our step counter 

        # After each episode, we decrease our epsilon until it gets to the set end, by passing in our decay parameters.
        epsilon = max(decayFunction(epsilon=epsilon, epsilonDecay=epsilonDecay, type=decayParam[2]), 0)

        # Once epsilon = 0, we are greedy (exploting), not exploring. 
        if(epsilon == 0):
            # We reduce our learning rate, to stabalize the Q values.
            alpha = 0.0001

        # Adds a reward update to our episode matrix. 
        if reward == 1:
            rewardsPerEpisode[i] = 1
            successfulEpisodes += 1
            stepsToReachGoal.append(steps) # Will push the number of steps taken to reach the goal for the specific episode. 
        
        # Calculate average reward per episode and store it in the array at index i.
        avgRewardPerEp[i] = np.mean(rewardsPerEpisode[:i+1])

    # Close the environment.
    env.close()

    # Calculate the percentage of successful episodes.
    percentageOfSuccessfulEpisodes = (successfulEpisodes / episodes) * 100

    # Calculate the average steps taken to reach the goal; checks if the list is empty.
    avgStepsToReachGoal = np.mean(stepsToReachGoal) if stepsToReachGoal else 0

    # Print evaluation metrics.
    # print(f"Average Reward per Episode: {avgRewardPerEp[-1]}")
    # print(f"Percentage of Successful Episodes: {percentageOfSuccessfulEpisodes}%")
    # print(f"Average Steps to Reach Goal: {avgStepsToReachGoal}")

    # Return the average total rewards per episode, for plotting.
    return avgRewardPerEp

# Used to run the Double-Q-Learning method.
if __name__ == '__main__':
    
    # Run the three function.
    avgTotalReward_linear = doubleQ(20000, decayParam=(1, 0.0001, "linear"))
    avgTotalReward_exponential = doubleQ(20000, decayParam=(1, 0.0001, "exponential"))
    avgTotalReward_inverse = doubleQ(20000, decayParam=(1, 0.0001, "inverse"))

    # Plot the average total reward for each decay function
    plt.figure(figsize=(10, 6))
    plt.plot(avgTotalReward_linear, label='Linear Decay')
    # plt.plot(avgTotalReward_exponential, label='Exponential Decay')
    # plt.plot(avgTotalReward_inverse, label='Inverse Decay')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Total Reward')
    plt.title('Average Total Reward vs. Number of Episodes for Different Decay Functions')
    plt.legend()
    plt.savefig("RL Methods/Double-Q-learning/Double-Q-Learning.png")
    plt.show()