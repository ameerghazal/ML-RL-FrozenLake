# Begin by importing gymnasium.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Run function.
def run(episodes, render=False):
    # actions: 0=left, 1=down, 2=right, 3=up
    # rewards: Reach goal=+1, Reach hole=0, Reach frozen=0

    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human" if render else None)

    #initializes an array 16 x 4 with zeroes
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    #alpha or learning rate
    learningRate = 0.9
    #gamma or discount factor
    gamma = 0.9


    #100% random actions
    epsilon = 1
    #epsilon decay rate | 1/0.0001 = 10,000
    epsilon_decay = 0.0001
    randomNum = np.random.default_rng()

    # For graph
    rewardsPerEpisode = np.zeros(episodes)

    for i in range(episodes):
        #states 0:15
        state = env.reset()[0]
        #True when agent falls in hole or reaches the goal
        terminated = False
        #True if steps is greater than 100
        truncated = False


        while(not terminated and not truncated):
            if randomNum.random() < epsilon:
                action = env.action_space.sample() #generate random action | actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = np.argmax(Q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            Q[state, action] = Q[state, action] + learningRate * (reward + gamma * np.max(Q[new_state,:]) - Q[state, action])

            state = new_state

        #after each episode decrease epsilon till it reaches zero
        epsilon = max(epsilon - epsilon_decay, 0)

        if(epsilon == 0):
            learningRate = 0.0001

        if reward == 1:
            rewardsPerEpisode[i] = 1

    env.close()


    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewardsPerEpisode[max(0,t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel("Episodes: Iterations")
    plt.ylabel("Sum of rewards during episodes")
    plt.savefig("QLearningFrozenLake.png")



if __name__ == '__main__':
    run(15000)