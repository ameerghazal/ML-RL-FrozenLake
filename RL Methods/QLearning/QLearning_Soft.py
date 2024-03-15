import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Soft-policy method
def softPolicy(epsilon, Q, state, env):
    # Check if the epslion is 0, if so, grab action from the Q-table.
    if epsilon < 0.001: return np.argmax(Q[state, :])

    # Calculate for the top part of the exp.
    expValues = np.exp(Q[state,:] - np.max(Q[state,:]))

    # Calculate the action probabilites:
    actionProb = np.round((expValues) / np.sum(expValues), 10)

    # Return the soft-policy action, based on the random action & probabilites.
    return np.random.choice(env.action_space.n, p=actionProb) 

# Decay function to test for different functions.
def decayFunction(epsilon, epsilonDecay, type="linear"):
    if (type == "exponetial"): # Exponential decay.
      return epsilon * np.exp(-epsilonDecay)
    
    if (type == "inverse"): # Inverse decay.
      return epsilon / (1 + epsilonDecay)
    
    return epsilon - epsilonDecay # Last case, return linear-step decay.


# Run function, in which we pass in the number of episodes, type of greedy policy, and if we want it to be rendered.
def normalQ(episodes, decayParam = (1, 0.0001, "linear"), policy = "greedy-policy", render=False):
    # actions: 0=left, 1=down, 2=right, 3=up
    # rewards: Reach goal=+1, Reach hole=0, Reach frozen=0e

    # Create the map and store.
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human" if render else None)

    # Initializes an array 16 x 4 with zeroes; Q(s,a) for all state and actions = 0
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # Alpha or learning rate or step-size (high alpha results in faster learning with potential oscillations, small alpha can result in more stability but slower learning).
    alpha = 0.5

    # gamma or discount factor
    gamma = 0.9

    #100% random actions; epsilon is the value that determines exploration vs. explotation.
    epsilon = decayParam[0]
    epsilonDecay = decayParam[1] #epsilon decay rate | 1/0.0001 = 10,000
    randomNum = np.random.default_rng() # Random number, for the conditonal.

    # For graph: intialize the rewards, epsilons to 0.
    rewardsPerEpisode = np.zeros(episodes)
    epsilonPerEp = np.zeros(episodes)
    avgRewardPerEp = np.zeros(episodes) # Initalize list of zeros, holding the average reward per epiosde.
    successful_episodes = 0  # Number of successful episodes, where the agent reaches the goal.
    steps_to_reach_goal = [] # List containing the number of steps to reach the goal per episode. 

    for i in range(episodes):
        #states 0:15
        state = env.reset()[0]
        #True when agent falls in hole or reaches the goal
        terminated = False
        #True if steps is greater than 100
        truncated = False
        # Counter for the number of steps in the episode, used for the average later.
        steps = 0

        # This entire iteration is an episode.,=
        while(not terminated and not truncated):
            
            # Call the policy, based on what was passed in.
            if (policy == "soft-policy"):
                action = softPolicy(epsilon, Q, state, env)
            elif (policy == "greedy-policy"): 
                # Check if a random number (between 0-0.99) is less than epsilon. # If so, we take a random action (explore).
                if randomNum.random() < epsilon:
                    action = env.action_space.sample() #generate random action | actions: 0=left, 1=down, 2=right, 3=up
                else:
                    action = np.argmax(Q[state,:]) # Otherwise, we take a greedy action based on the data we have from Q matrix (exploit).
                
            # Take action A, observe reward and next state.
            new_state,reward,terminated,truncated,_ = env.step(action)
            
            # Q(s,a) = Q(s,a) + alpha(Reward + gamma * max(Q(s',a)) - Q(s,a))
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state,:]) - Q[state, action])

            # S = S'
            state = new_state

            # Increment our step counter 
            steps += 1

        # Note the epsilon used for the episode.     
        epsilonPerEp[i] = epsilon

        # After each episode, we decrease our epsilon until it gets to 0.
        epsilon = max(decayFunction(epsilon=epsilon, epsilonDecay=epsilonDecay, type=decayParam[2]), 0)

        # Once epsilon = 0, we are greedy (exploting), not exploring. 
        if(epsilon == 0):
            # We reduce our learning rate, to stabalize the Q values.
            alpha = 0.0001

        # Adds a reward update to our episode matrix. 
        if reward == 1:
            rewardsPerEpisode[i] = 1 
            successful_episodes += 1
            steps_to_reach_goal.append(steps) # Will push the number of steps taken to reach the goal for the specific episode. 
        
        # Calculate average reward per episode and store it in the array at index i.
        avgRewardPerEp[i] = np.mean(rewardsPerEpisode[:i+1])

    # Close the environment.
    env.close()

    # Calculate the percentage of successful episodes.
    percentage_successful_episodes = (successful_episodes / episodes) * 100

    # Calculate the average steps taken to reach the goal; checks if the list is empty.
    average_steps_to_reach_goal = np.mean(steps_to_reach_goal) if steps_to_reach_goal else 0

    # Print evaluation metrics.
    # print(f"Average Reward per Episode: {avgRewardPerEp[-1]}")
    # print(f"Percentage of Successful Episodes: {percentage_successful_episodes}%")
    # print(f"Average Steps to Reach Goal: {average_steps_to_reach_goal}")

    return avgRewardPerEp

# Used to run the Q-learning method.
if __name__ == '__main__':
    avgTotalRewardSoft = normalQ(15000, policy="soft-policy") # Soft-max policy return.
    avgTotalRewardGreedy = normalQ(15000, policy="greedy-policy") # Greedy-max policy return.

    # Plot the average total reward for each decay function
    plt.figure(figsize=(10, 6))
    plt.plot(avgTotalRewardSoft, label='Softmax Policy')
    plt.plot(avgTotalRewardGreedy, label='Greedy Policy')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Total Reward')
    plt.title('Average Total Reward vs. Number of Episodes for Q-Learning Softmax vs. Greedy Polices')
    plt.legend()
    plt.savefig("RL Methods/Q-Learning/soft-max-policy.png")
    plt.show()