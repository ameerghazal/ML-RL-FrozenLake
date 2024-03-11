# Begin by importing gymnasium.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Run function, in which we pass in the number of episodes.
def run(episodes, render=False):
    # actions: 0=left, 1=down, 2=right, 3=up
    # rewards: Reach goal=+1, Reach hole=0, Reach frozen=0e

    # Create the map and store.
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human" if render else None)

    # Small epsilon paramter.
    epsilon = 0.1

    # Gamma or discount factor
    gamma = 0.9

    # **Initialize policy pi(s,a) 16x4 for all state, actions = 1 (arbitary-epsilon-policy)
    pi = np.ones((env.observation_space.n, env.action_space.n)) * epsilon / env.action_space.n  # ε-soft policy

    # Initalize Q(s,a) 16x4 for all state, actions arbitarly to encourage explortation.
    Q = np.random.rand(env.observation_space.n, env.action_space.n) * 0.01

    # Initalize Returns(s,a); Empty list based on (s,a) pair, and the return G.
    returns = {(s,a): [] for s in range(env.observation_space.n) for a in range(env.action_space.n)}

    # Initialize success rate, average steps, and average reward per episode
    success_rate = 0
    avg_steps_to_goal = 0
    avg_reward_per_episode = 0

    # Optimal path
    optimal_path = []

    # Loop for each episode 'forever'.
    for _ in range(episodes):
        episode = [] # Init an empty list for the episode.
        state = env.reset()[0] # Starting state at top of grid.
        #True when agent falls in hole or reaches the goal
        terminated = False
        #True if steps is greater than 100
        truncated = False
        # Counts the number of steps per episode.
        steps = 0

        while not terminated and not truncated: # Loop until the episode is over.

            # Check if a random number (between 0-0.99) is less than epsilon. # If so, we take a random action (explore).
            if np.random.default_rng().random() < epsilon:
                action = env.action_space.sample() #generate random action | actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = np.argmax(Q[state,:]) # Otherwise, we take a greedy action based on the data we have from Q matrix (exploit). 

            # Take action A, observe reward, next state, and determine if the run is over.
            new_state,reward,terminated,truncated,_ = env.step(action)
            
            # Observe the state action and reward in our episode.
            episode.append((state, action, reward))

            # Update the state to the next state.
            state = new_state

            # Update the amount of steps taken.
            steps += 1

        # Update the averages.
        avg_steps_to_goal += steps
        avg_reward_per_episode += reward

        # Update the success-rate if the goal is reached.
        if reward == 1:
            success_rate += 1
            # Store the optimal path
            optimal_path = episode

        # Set the return equal to 0.
        G = 0

        # Loop for each step of the episode in reversed order.
        for step in reversed(episode):

            # Get the state, action, and reward.
            state, action, reward = step

            # Update the return, based on the reward.
            G = (gamma * G) + reward

            # If the pair (S, A) does not appear in sequence S0, A0, S1, A1, ..., St-1, At-1, then we do the following. We pull the (state, action) pair from the episode list exluding the final episode. We then check if the current (state, action) is in the updated list. If not, do the seqeunce. Otherwise, skip.
            if (state, action) not in [(x[0], x[1]) for x in episode[:-1]]:

                # Append G to Returns(St, At)
                returns[(state, action)].append(G)

                # Update Q(st, At) <- average(Returns(St, At))
                Q[state, action] = np.mean(returns[((state, action))])

                # Find the optimal action, A* <- argmaxQ(St, a)
                optimalAction = np.argmax(Q[state, :])
                
                # For all actions in A(St): pi(a | St) = 
                # if (a = A*) 1 - epsilon + epsilon / abs(A(st))
                # else epsilon / abs(A(St))
                for a in range(env.action_space.n):
                    if a == optimalAction:
                        pi[state, a] = 1 - epsilon + (epsilon / env.action_space.n) # Greedy Action.
                    else:
                        pi[state, a] = epsilon / env.action_space.n # Non-greedy Action.

    # Average everything out.
    success_rate /= episodes
    avg_steps_to_goal /= episodes
    avg_reward_per_episode /= episodes

    # Print out everything.
    print("Success Rate:", success_rate)
    print("Average Steps to Reach Goal:", avg_steps_to_goal)
    print("Average Reward per Episode:", avg_reward_per_episode)            
    
    # Return the final Q-values and policy
    print("Q-values:")    
    print(Q)
    print("Policy:")
    print(pi)

    return Q, pi, optimal_path


# Used to run the Q-learning method.
if __name__ == '__main__':
    Q, pi, optimal_path = run(10000, render=False)

    print(optimal_path)

    # Render the optimal path
    if optimal_path:
        print("Rendering optimal path:")
        env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,    render_mode="human")
        state = env.reset()[0]
        terminated = False
        while not terminated:
            action = np.argmax(Q[state, :])
            print(action)
            new_state,reward,terminated,_,_ = env.step(action)
            # Update the state to the next state.
            state = new_state
            env.render()


  # For example, if during an episode the agent visits state s and takes action a, resulting in a return of G, the return value G is appended to the list Returns[(s, a)]. This allows the agent to later update its estimate of the expected return for that state-action pair based on the observed returns. This is crucial for the Monte Carlo method to learn the optimal policy by averaging the returns over multiple episodes.




# import 


