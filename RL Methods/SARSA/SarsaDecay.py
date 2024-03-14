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
def run(episodes, decayParam = (1, 0.0001, "linear"), render=False):
    # actions: 0=left, 1=down, 2=right, 3=up
    # rewards: Reach goal=+1, Reach hole=0, Reach frozen=0

    # Create the map and store.
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human" if render else None)

    # Initializes an array 16 x 4 with zeroes; Q(s,a) for all state and actions = 0
    Q = np.zeros((env.observation_space.n, env.action_space.n))

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
    epsilonPerEp = np.zeros(episodes)
    avgRewardPerEp = np.zeros(episodes) # Initalize list of zeros, holding the average reward per epiosde.
    successfulEpisodes = 0  # Number of successful episodes, where the agent reaches the goal.
    stepsToReachGoal = [] # List containing the number of steps to reach the goal per episode. 

    for i in range(episodes):
        state = env.reset()[0] #states 0:15
        terminated = False #True when agent falls in hole or reaches the goal
        truncated = False  #True if steps is greater than 100
        
        # Counter for the number of steps in the episode, used for the average later.
        steps = 0

        # Check if a random number (between 0-0.99) is less than epsilon. # If so, we take a random action (explore).
        if randomNum.random() < epsilon:
          action = env.action_space.sample() #generate random action | actions: 0=left, 1=down, 2=right, 3=up
        else:
          action = np.argmax(Q[state,:]) # Otherwise, we take a greedy action based on the data we have from Q matrix (exploit). 

        # This entire iteration is an episode.
        while(not terminated and not truncated):
      
            # Take action A, observe reward and next state.
            nextState,reward,terminated,truncated,_ = env.step(action)

            # Check if a random number (between 0-0.99) is less than epsilon. # If so, we take a random action (explore).
            if randomNum.random() < epsilon:
                nextAction = env.action_space.sample() #generate random action | actions: 0=left, 1=down, 2=right, 3=up
            else:
                nextAction = np.argmax(Q[nextState,:]) # Otherwise, we take a greedy action based on the data we have from Q matrix (exploit). 

            # Q(s,a) = Q(s,a) + alpha(Reward + gamma * max(Q(s',a)) - Q(s,a))
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[nextState, nextAction] - Q[state, action])

            # S = S', A = A'
            state = nextState
            action = nextAction
            steps += 1 # Increment our step counter 


        # Note the epsilon used for the episode.     
        epsilonPerEp[i] = epsilon

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
    print(f"Average Reward per Episode: {avgRewardPerEp[-1]}")
    print(f"Percentage of Successful Episodes: {percentageOfSuccessfulEpisodes}%")
    print(f"Average Steps to Reach Goal: {avgStepsToReachGoal}")

    # Return the average total rewards per episode, for plotting.
    return avgRewardPerEp

# Used to run the SARSA-Decay method.
if __name__ == '__main__':
    
    # Run the three function.
    avgTotalReward_linear = run(20000, decayParam=(1, 0.0001, "linear"))
    avgTotalReward_exponential = run(20000, decayParam=(1, 0.0001, "exponential"))
    avgTotalReward_inverse = run(20000, decayParam=(1, 0.0001, "inverse"))

    # Plot the average total reward for each decay function
    plt.figure(figsize=(10, 6))
    plt.plot(avgTotalReward_linear, label='Linear Decay')
    plt.plot(avgTotalReward_exponential, label='Exponential Decay')
    plt.plot(avgTotalReward_inverse, label='Inverse Decay')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Total Reward')
    plt.title('Average Total Reward vs. Number of Episodes for Different Decay Functions')
    plt.legend()
    plt.savefig("RL Methods/SARSA/SARSADecay.png")
    plt.show()

'''
    # Evaluate policy
    total_reward = 0
    num_episodes = 100
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state, :])
            state, reward, done, _ = env.step(action)
            total_reward += reward

    print("Average reward:", total_reward / num_episodes)
'''
































# # Used to smooth out the graph.
# def moving_average(data, window_size):
#     cumsum = np.cumsum(data, dtype=float)
#     cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
#     return cumsum[window_size - 1:] / window_size

# # Function to run multiple episodes
# def run_multiple_runs(runs, episodes_per_run, inverse_decay, decay_params):
#     rewards_per_episode_all_runs = [] 
#     epsilon_per_episode = []
#     successful_episodes_all_runs = []
#     steps_to_reach_goal_per_episode = []

#     for _ in range(runs):
#         rewards_per_episode, epsilon_per_episode, successful_episodes, steps_to_reach_goal  = run(episodes_per_run, inverse_decay, decay_params)
#         rewards_per_episode_all_runs.append(rewards_per_episode)
#         successful_episodes_all_runs.append(successful_episodes )
#         steps_to_reach_goal_per_episode.append(steps_to_reach_goal )


#     # Calculate the average rewards per episode over all runs
#     avg_rewards_per_episode = np.mean(rewards_per_episode_all_runs, axis=0)

#     # Smooth the curves with a moving average
#     smooth_rewards_per_episode = moving_average(avg_rewards_per_episode, window_size=100)

#     # Calculate the overall percentage of successful episodes
#     total_episodes = runs * episodes_per_run
#     total_successful_episodes = sum(successful_episodes_all_runs)
#     percentage_successful_episodes = (total_successful_episodes / total_episodes) * 100

#     # Plot the smoothed curves
#     fig, ax1 = plt.subplots()

#     # Plot smoothed rewards on the first y-axis
#     ax1.plot(epsilon_per_episode, label='Epsilon Values', color='r')
#     ax1.set_xlabel('Episodes')
#     ax1.set_ylabel('Epsilon Values', color='r')
#     ax1.tick_params('y', colors='r')

#     # Create a second y-axis
#     ax2 = ax1.twinx()
#     ax2.plot(smooth_rewards_per_episode, label='Average Rewards per Episode', color='b')
#     ax2.set_ylabel('Average Rewards per Episode', color='b')
#     ax2.tick_params('y', colors='b')

#     # Combine legends from both axes
#     lines, labels = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines + lines2, labels + labels2, loc='upper center')


#     plt.title("SARSA Epsilon and Sum of Rewards vs. Episodes")
#     plt.show()
#     #plt.savefig("RL Methods/SARSA/SARSAEpsilon.png")

#     # Print the evaluation metrics after all runs are completed
#     # print(f"Average Reward per Episode: {np.mean(avg_rewards_per_episode)}")
#     # print(f"Percentage of Successful Episodes: {percentage_successful_episodes}%")
#     # print(f"Average Steps to Reach Goal: {np.mean(steps_to_reach_goal_per_episode)}")

#     # Return the following.
#     return avg_rewards_per_episode, epsilon_per_episode

# def linear_decay(epsilon_start, epsilon_end, epsilon_decay):
#     return max(epsilon_start - epsilon_decay, epsilon_end)

# def exponential_decay(epsilon_start, epsilon_decay):
#     return epsilon_start * np.exp(-epsilon_decay)

# def inverse_decay(epsilon_start, epsilon_decay):
#     return epsilon_start / (1 + epsilon_decay)

# def cosine_decay(epsilon_start, epsilon_end, epsilon_decay):
#     return epsilon_end + (epsilon_start - epsilon_end) * (1 + np.cos(np.pi / epsilon_decay)) / 2

# def piecewise_decay(episode):
#     if episode < 1000:
#         return 1.0
#     elif episode < 2000:
#         return 0.5
#     else:
#         return 0.1


# def run(episodes, decay_function, decay_params, render=False):
#     env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human" if render else None)
#     Q = np.zeros((env.observation_space.n, env.action_space.n))
#     alpha = 0.5
#     gamma = 0.9
#     epsilon = decay_function(*decay_params)  # Initialize epsilon using decay function
#     randomNum = np.random.default_rng()
#     rewardsPerEpisode = np.zeros(episodes)
#     epsilonPerEp = np.zeros(episodes)
#     avgRewardPerEp = np.zeros(episodes)
#     successful_episodes = 0
#     steps_to_reach_goal = []

#     for i in range(episodes):
#         state = env.reset()[0]
#         terminated = False
#         truncated = False
#         steps = 0

#         while not terminated and not truncated:
#             action = np.argmax(Q[state, :]) if randomNum.random() >= epsilon else env.action_space.sample()

#             new_state, reward, terminated, truncated, _ = env.step(action)

#             next_action = np.argmax(Q[new_state, :]) if randomNum.random() >= epsilon else env.action_space.sample()

#             Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[new_state, next_action] - Q[state, action])

#             state = new_state
#             steps += 1

#         epsilonPerEp[i] = epsilon
#         epsilon = decay_function(epsilon, decay_params[1], decay_params[2])

#         if epsilon == 0:
#             alpha = 0.0001

#         if reward == 1:
#             rewardsPerEpisode[i] = 1
#             successful_episodes += 1
#             steps_to_reach_goal.append(steps)

#         avgRewardPerEp[i] = np.mean(rewardsPerEpisode[:i + 1])

#     env.close()

#     percentage_successful_episodes = (successful_episodes / episodes) * 100
#     average_steps_to_reach_goal = np.mean(steps_to_reach_goal) if steps_to_reach_goal else 0

#     # print(f"Average Reward per Episode: {avgRewardPerEp[-1]}")
#     # print(f"Percentage of Successful Episodes: {percentage_successful_episodes}%")
#     # print(f"Average Steps to Reach Goal: {average_steps_to_reach_goal}")

#     sum_rewards = np.zeros(episodes)
#     for t in range(episodes):
#         sum_rewards[t] = np.sum(rewardsPerEpisode[max(0, t - 100):(t + 1)])

    
#     return rewardsPerEpisode, epsilonPerEp, percentage_successful_episodes, steps_to_reach_goal

# if __name__ == '__main__':
#     # Example usage with linear decay function
#     decay_params = (1, 0, 0.05)  # epsilon_start, epsilon_end, epsilon_decay
#     run_multiple_runs(50, 10000, linear_decay, decay_params)



