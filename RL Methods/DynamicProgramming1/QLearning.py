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

    # Initializes an array 16 x 4 with zeroes; Q(s,a) for all state and actions = 0
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # alpha or learning rate or step-size
    alpha = 0.9

    # gamma or discount factor
    gamma = 0.9


    #100% random actions
    epsilon = 1
    #epsilon decay rate | 1/0.0001 = 10,000
    epsilon_decay = 0.0001
    randomNum = np.random.default_rng()

    # For graph; intialize the rewards in every episode to 0.
    rewardsPerEpisode = np.zeros(episodes)

    for i in range(episodes):

        #states 0:15
        state = env.reset()[0]
        #True when agent falls in hole or reaches the goal
        terminated = False
        #True if steps is greater than 100
        truncated = False

        # This entire iteration is an episode.,=
        while(terminated or truncated):
            # Check if a random number (between 0-0.99) is less than epsilon. # If so, we take a random action (explore).
            if randomNum.random() < epsilon:
                action = env.action_space.sample() #generate random action | actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = np.argmax(Q[state,:]) # Otherwise, we take a greedy action based on the data we have from Q matrix. 

            # Take actuin A, observe reward and next state.
            new_state,reward,terminated,truncated,_ = env.step(action)
            
            # Q(s,a) = Q(s,a) + alpha(Reward + gamma * max(Q(s',a)) - Q(s,a))
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state,:]) - Q[state, action])

            # S = S'
            state = new_state

        # After each episode, we decrease our epsilon until it gets to 0.
        epsilon = max(epsilon - epsilon_decay, 0)

        # Once epsilon = 0, we are greedy (exploting), not exploring. 
        if(epsilon == 0):
            # We reduce our learning rate, to stabalize the Q values.
            alpha = 0.0001

        # Adds a reward update to our episode matrix. 
        if reward == 1:
            rewardsPerEpisode[i] = 1

    env.close()


    # Plot of the rewards vs. Episodes; running sum of the rewards of every 100 episodes.
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewardsPerEpisode[max(0,t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel("Episodes: Iterations")
    plt.ylabel("Sum of rewards during episodes")
    plt.savefig("QLearningFrozenLake.png")


# Used to run the Q-learning method.
if __name__ == '__main__':
    run(15000)