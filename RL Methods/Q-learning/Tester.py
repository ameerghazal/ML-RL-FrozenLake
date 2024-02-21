# Begin by importing gymnasium.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def run(episodes, render=False):
    # actions: 0=left, 1=down, 2=right, 3=up
    # rewards: Reach goal=+1, Reach hole=0, Reach frozen=0

    # Create the map and store.
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human" if render else None)

    # Initializes an array 16 x 4 with zeroes; Q(s,a) for all state and actions = 0
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # alpha or learning rate or step-size
    alpha = 0.9

    # gamma or discount factor
    gamma = 0.9

    # 100% random actions; epsilon is the value that determines exploration vs. exploitation.
    epsilon = 1
    # epsilon decay rate | 1/0.0001 = 10,000
    epsilon_decay = 0.0001
    randomNum = np.random.default_rng()

    # For graph; initialize the rewards in every episode to 0.
    rewardsPerEpisode = np.zeros(episodes)

    # Initialize evaluation metrics
    average_reward_per_episode = np.zeros(episodes)
    successful_episodes = 0
    steps_to_reach_goal = []

    # Initialize epsilon values for plotting
    epsilon_values = np.linspace(1, 0, episodes)

    for i in range(episodes):

        # states 0:15
        state = env.reset()[0]
        # True when agent falls in hole or reaches the goal
        terminated = False
        # True if steps is greater than 100
        truncated = False
        steps = 0

        # This entire iteration is an episode.
        while(not terminated and not truncated):
            # Check if a random number (between 0-0.99) is less than epsilon.
            # If so, we take a random action (explore).
            if randomNum.random() < epsilon:
                action = env.action_space.sample() # generate random action | actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = np.argmax(Q[state,:]) # Otherwise, we take a greedy action based on the data we have from Q matrix (exploit). 

            # Take action A, observe reward and next state.
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Update Q(s,a) using the Q-learning update rule.
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state,:]) - Q[state, action])

            # Update current state.
            state = new_state

            # Increment step count.
            steps += 1

        # Update epsilon for epsilon-greedy policy.
        epsilon = max(epsilon - epsilon_decay, 0)

        # Once epsilon = 0, we are greedy (exploiting), not exploring. 
        if(epsilon == 0):
            # Reduce learning rate to stabilize Q values.
            alpha = 0.0001

        # Update rewards per episode.
        if reward == 1:
            rewardsPerEpisode[i] = 1
            successful_episodes += 1
            steps_to_reach_goal.append(steps)

        # Calculate average reward per episode.
        average_reward_per_episode[i] = np.mean(rewardsPerEpisode[:i+1])

    # Close the environment.
    env.close()

    # Calculate percentage of successful episodes.
    percentage_successful_episodes = (successful_episodes / episodes) * 100

    # Calculate average steps taken to reach the goal.
    average_steps_to_reach_goal = np.mean(steps_to_reach_goal) if steps_to_reach_goal else 0

    # Plot rewards and epsilon decay
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewardsPerEpisode[max(0,t-100):(t+1)])
    plt.plot(sum_rewards, label='Sum of rewards during episodes')
    plt.plot(epsilon_values, label='Epsilon decay', linestyle='--')
    plt.xlabel("Episodes: Iterations")
    plt.ylabel("Rewards/Epsilon")
    plt.legend()
    plt.show()

    # Return evaluation metrics.
    return average_reward_per_episode, percentage_successful_episodes, average_steps_to_reach_goal, epsilon_values

# Used to run the Q-learning method and evaluate its performance.
if __name__ == '__main__':
    episodes = 15000
    average_reward_per_episode, percentage_successful_episodes, average_steps_to_reach_goal, epsilon_values = run(episodes)
    

    # Plot rewards and epsilon decay
    plt.figure(figsize=(10, 6))
    plt.plot(average_reward_per_episode, label='Average Reward per Episode')
    plt.plot(epsilon_values)
    plt.xlabel("Episodes: Iterations")
    plt.ylabel("Average Reward per Episode")
    plt.ylim(0, 1)  # Set y-axis limits between 0 and 1
    plt.legend()
    plt.show()


    # Print evaluation metrics.
    print(f"Average Reward per Episode: {average_reward_per_episode[-1]}")
    print(f"Percentage of Successful Episodes: {percentage_successful_episodes}%")
    print(f"Average Steps to Reach Goal: {average_steps_to_reach_goal}")
