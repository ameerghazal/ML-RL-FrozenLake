# Begin by importing gymnasium.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Plotting the heat-map, V-values, and optimal policy.
def plot_values_and_policy(V, policy, env, rewardFunction = "default"):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plotting the Optimal Value and Optimal Policy.
    im = axs[0].imshow(V.reshape(4, 4), cmap='Greens', interpolation='nearest')
    for i in range(4):
        for j in range(4):
            # State labels
            if env.unwrapped.desc[i, j] == b'S':
                axs[0].text(j, i, 'Start', ha='center', va='center', color='black')
            elif env.unwrapped.desc[i, j] == b'F':
                axs[0].text(j, i, 'Frozen', ha='center', va='center', color='black')
            elif env.unwrapped.desc[i, j] == b'H':
                axs[0].text(j, i, 'Hole', ha='center', va='center', color='black')
            elif env.unwrapped.desc[i, j] == b'G':
                axs[0].text(j, i, 'Goal', ha='center', va='center', color='black')

            # Value function
            axs[0].text(j, i + 0.2, f'{V[i * 4 + j]:.2f}', ha='center', va='center', color='black')
    axs[0].set_title('Optimal Value Function (V*)')

    # Create a colorbar to show the value scale
    cbar = fig.colorbar(im, ax=axs[0], orientation='horizontal', pad=0.1)
    cbar.set_label('Relative Reward (Higher is better)', rotation=0, labelpad=10)

    # Plot Policy pi*
    axs[1].imshow(np.ones((4, 4)), cmap='Blues', interpolation='nearest', )
    for i in range(4):
        for j in range(4):
            action = policy[i * 4 + j]
            dx, dy = 0, 0
            box_width = 1.0 / 4
            box_height = 1.0 / 4
            if action == 0:  # left
                dx = -box_width / 2
            elif action == 1:  # down
                dy = box_height / 2
            elif action == 2:  # right
                dx = box_width / 2
            elif action == 3:  # up
                dy = -box_height / 2
            axs[1].arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.2, head_length=0.2, fc='black', ec='black')
            axs[1].text(j + 0.5, i + 0.5, f"{action}", ha='center', va='center', color='black', fontsize=14)

    # Add grid lines to the policy plot
    axs[1].set_xticks(np.arange(0, 5, 1))
    axs[1].set_yticks(np.arange(0, 5, 1))
    axs[1].set_xlim(0, 4)  # Set x-axis limits
    axs[1].set_ylim(4, 0)  # Set y-axis limits
    axs[1].grid(True, which='both', color='gray', linestyle='-', linewidth=1)
    axs[1].set_title('Optimal Policy (pi*)')

    # Save the figure based on the type of reward function.
    if rewardFunction == "custom":
      plt.savefig("RL Methods/Dynamic-Programming/Negative-Reward.png")
    else:
      plt.savefig("RL Methods/Dynamic-Programming/Normal-Reward.png")

    plt.show()

# Custom reward function.
def customRewardFunction(state, env):
    row, col = np.unravel_index(state, (4, 4))
    if state in [5, 7, 11, 12]:  # Hole states
        return -1  # Penalty for falling into a hole
    elif state == 15:  # Goal state
        return 1 # Reward for goal
    else:
        return -0.5 # Default penalty for other states

# Run function, in which we pass in the number of episodes.
def run(episodes, rewardFunction="default", render=False):
    # actions: 0=left, 1=down, 2=right, 3=up
    # rewards: Reach goal=+1, Reach hole=0, Reach frozen=0

    # Create the map and store.
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human" if render else None)

    # Initializes an array 16 with all 16 states.
    V = np.zeros(env.observation_space.n)

    # gamma or discount factor
    gamma = 0.9

    # Small threshold theta for determining the accuracy.
    theta = 1e-10

    # Iterations to compare.
    iterations = 0

    # Outer loop, goes until stopped.
    while True:

        # Reinitalize the difference to 0.
        delta = 0

        # Loop for each state in the list of states.
        for state in range(env.observation_space.n):
            # Setting the local value to the V(s) pair. For example, if 12 is state, then val = V(12)
            val = V[state]

            # Find the negative reward based on the index on the grid.
            # Assuming the goal is at (3, 3)

            # For each state, the V(s) is updated to the max across all possible actions. Hence, we loop for each action and use the max function. We check the max by summing the product of the transition probability, reward, and discounted value of the next-state, which we grab these values from the enviroment given our current state and current action. We add the negative reward, as well.
            V[state] = max(sum([probability * (
                        (customRewardFunction(state, env) if rewardFunction == "custom" else reward) + gamma *
                        V[nextState]) for probability, nextState, reward, _ in env.unwrapped.P[state][action]]) for
                           action in range(env.action_space.n))

            # Calculate the difference for the next iteration.
            delta = max(delta, abs(val - V[state]))

        # Increment for comparison
        iterations += 1

        # Conditional to check if we are done.
        if (delta < theta): break

    # Output a deterministic policy
    policy = np.zeros(env.observation_space.n,
                      dtype=int)  # Initlaize an array of size 16, with a type integer for the policy of each state.
    for state in range(env.observation_space.n):  # From there, we loop for each state again.

        # The policy, given the state, chooses the action index that maximizes the sum of the products of the probailites, rewards, and the discounted next state. This is done for each possible next state resulting from taking different actions in the current state. The np.argmax function is then used to find the index corresponding to the action that yields the highest sum.
        policy[state] = np.argmax([sum([probability * (
                    (customRewardFunction(state, env) if rewardFunction == "custom" else reward) + gamma * V[
                nextState]) for probability, nextState, reward, _ in env.unwrapped.P[state][action]]) for action in
                                   range(env.action_space.n)])

    # Print the policy and V.
    print(f"Optimal Policy: \n {policy.reshape(4, 4)}")  # Remember, we are using isSlippery.
    print(f"Value Function: \n {V.reshape(4, 4)}")
    print(f"Iterations done:\n {iterations}")

    # # Render the environment with the optimal policy
    # state = env.reset()[0]  # Starts at position 0.
    # env.render()  # Renders the map.
    # terminated = False  # Checks if the agent has fallen reached the goal.
    # while not terminated:
    #     action = policy[state]  # Given the state, pass it into the policy function and return the correct action.
    #     state, reward, terminated, _, _ = env.step(
    #         action)  # Take the action and return the obs. state, reward, if terminated, and more.
    #     env.render()  # Render the new move.

    # Plot the policy and state value.
    plot_values_and_policy(V, policy, env, rewardFunction)

# Used to run the Dynamic method.
if __name__ == '__main__':
    run(1, rewardFunction="default") # Use the original reward function.
    run(1, rewardFunction="custom") # Use the custom negative reward function.

'''
Remember, when rendering, the optimal policy is like a grid.
For example, if [1 2 1 0 1 0 1 0 2 1 1 0 0 2 2 0] is returned, it is in the index order of a typical unit indexed array [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]. That is, if we pass in the state 1 into our policy function (policy[1]), output will be the best action, which in our case is 1 (down).
Another example would be Policy[4], which returns 1 (down).
'''