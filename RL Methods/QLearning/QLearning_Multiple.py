# Import required libraries.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Used to smooth out the graph.
def moving_average(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

# Function to run multiple episodes
def run_multiple_runs(runs, episodes_per_run):
    rewards_per_episode_all_runs = [] 
    epsilon_per_episode = []
    successful_episodes_all_runs = []
    steps_to_reach_goal_per_episode = []

    for _ in range(runs):
        rewards_per_episode, epsilon_per_episode, successful_episodes, steps_to_reach_goal  = run(episodes_per_run)
        rewards_per_episode_all_runs.append(rewards_per_episode)
        successful_episodes_all_runs.append(successful_episodes )
        steps_to_reach_goal_per_episode.append(steps_to_reach_goal )


    # Calculate the average rewards per episode over all runs
    avg_rewards_per_episode = np.mean(rewards_per_episode_all_runs, axis=0)

    # Smooth the curves with a moving average
    smooth_rewards_per_episode = moving_average(avg_rewards_per_episode, window_size=100)

    # Calculate the overall percentage of successful episodes
    total_episodes = runs * episodes_per_run
    total_successful_episodes = sum(successful_episodes_all_runs)
    percentage_successful_episodes = (total_successful_episodes / total_episodes) * 100

    # Plot the smoothed curves
    fig, ax1 = plt.subplots()

    # Plot smoothed rewards on the first y-axis
    ax1.plot(epsilon_per_episode, label='Epsilon Values', color='r')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Epsilon Values', color='r')
    ax1.tick_params('y', colors='r')

    # Create a second y-axis
    ax2 = ax1.twinx()
    ax2.plot(smooth_rewards_per_episode, label='Average Rewards per Episode', color='b')
    ax2.set_ylabel('Average Rewards per Episode', color='b')
    ax2.tick_params('y', colors='b')

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center')

    plt.title('Q-Learning Epsilon and Average Sum of Rewards per Episode over Multiple Runs')
    plt.show()

    # Print the evaluation metrics after all runs are completed
    # print(f"Average Reward per Episode: {np.mean(avg_rewards_per_episode)}")
    # print(f"Percentage of Successful Episodes: {percentage_successful_episodes}%")
    # print(f"Average Steps to Reach Goal: {np.mean(steps_to_reach_goal_per_episode)}")

    # Return the following.
    return avg_rewards_per_episode, epsilon_per_episode

# Run function, in which we pass in the number of episodes.
def run(episodes, render=False):
    # actions: 0=left, 1=down, 2=right, 3=up
    # rewards: Reach goal=+1, Reach hole=0, Reach frozen=0e

    # Create the map and store.
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,
                   render_mode="human" if render else None)

    # Initializes an array 16 x 4 with zeroes; Q(s,a) for all state and actions = 0
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # Alpha or learning rate or step-size (high alpha results in faster learning with potential oscillations, small alpha can result in more stability but slower learning).
    alpha = 0.5

    # gamma or discount factor
    gamma = 0.9

    # 100% random actions; epsilon is the value that determines exploration vs. explotation.
    epsilon = 1
    epsilon_decay = 0.001  # epsilon decay rate | 1/0.0001 = 10,000
    randomNum = np.random.default_rng()  # Random number, for the conditonal.

    # For graph: intialize the rewards, epsilons to 0.
    rewardsPerEpisode = np.zeros(episodes)
    epsilonPerEp = np.zeros(episodes)

    # Initalize list of zeros, holding the average reward per epiosde.
    avgRewardPerEp = np.zeros(episodes)
    successful_episodes = 0  # Number of successful episodes, where the agent reaches the goal.
    steps_to_reach_goal = []  # List containing the number of steps to reach the goal per episode.

    for i in range(episodes):

        # states 0:15
        state = env.reset()[0]
        # True when agent falls in hole or reaches the goal
        terminated = False
        # True if steps is greater than 100
        truncated = False
        # Counter for the number of steps in the episode, used for the average later.
        steps = 0

        # This entire iteration is an episode.,=
        while (not terminated and not truncated):
            # Check if a random number (between 0-0.99) is less than epsilon. # If so, we take a random action (explore).
            if randomNum.random() < epsilon:
                action = env.action_space.sample()  # generate random action | actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = np.argmax(Q[state, :])  # Otherwise, we take a greedy action based on the data we have from Q matrix (exploit).

            # Take action A, observe reward and next state.
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Q(s,a) = Q(s,a) + alpha(Reward + gamma * max(Q(s',a)) - Q(s,a))
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            # S = S'
            state = new_state

            # Increment our step counter
            steps += 1

        # Note the epsilon used for the episode.
        epsilonPerEp[i] = epsilon

        # After each episode, we decrease our epsilon until it gets to 0.
        epsilon = max(epsilon - epsilon_decay, 0)

        # Once epsilon = 0, we are greedy (exploting), not exploring.
        if (epsilon == 0):
            # We reduce our learning rate, to stabalize the Q values.
            alpha = 0.0001

        # Adds a reward update to our episode matrix.
        if reward == 1:
            rewardsPerEpisode[i] = 1
            successful_episodes += 1
            steps_to_reach_goal.append(steps)  # Will push the number of steps taken to reach the goal for the specific episode.

        # Calculate average reward per episode and store it in the array at index i.
        avgRewardPerEp[i] = np.mean(rewardsPerEpisode[:i + 1])

    # Close the environment.
    env.close()

    # Calculate the percentage of successful episodes.
    percentage_successful_episodes = (successful_episodes / episodes) * 100

    # Calculate the average steps taken to reach the goal; checks if the list is empty.
    average_steps_to_reach_goal = np.mean(steps_to_reach_goal) if steps_to_reach_goal else 0

    # Plot of the rewards vs. Episodes; running sum of the rewards of every 100 episodes.
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewardsPerEpisode[max(0, t - 100):(t + 1)])

    return rewardsPerEpisode, epsilonPerEp, percentage_successful_episodes, steps_to_reach_goal

# Used to run the Q-learning method.
if __name__ == '__main__':
    run_multiple_runs(1000, 1500)
