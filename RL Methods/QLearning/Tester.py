# Begin by importing gymnasium.
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_values_and_policy(V, policy, env):
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

    plt.savefig("RL Methods/Dynamic-Programming/Dynamic-Plots.png")
    plt.show()


# Run function, in which we pass in the number of episodes.
def run(episodes, render=False):
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

    V = np.zeros(env.observation_space.n)
    gamma = 0.9
    theta = 1e-10

    iterations = 0

    while True:
        delta = 0
        for state in range(env.observation_space.n):
            val = V[state]
            # Initialize the maximum value with a large negative number
            max_action_value = float('-inf')
            for action in range(env.action_space.n):
                # Calculate the value of the current state-action pair
                action_value = 0
                for probability, next_state, reward, _ in env.unwrapped.P[state][action]:
                    # Modify the rewards here
                    if reward > 0:  # If the reward is positive
                        action_value += probability * (reward + gamma * V[next_state])
                    else:  # If the reward is non-positive (including negative)
                        action_value += probability * (reward + gamma * V[next_state])
                # Update the maximum action value for the current state
                max_action_value = max(max_action_value, action_value)
            # Update the value function for the current state
            V[state] = max_action_value
            # Calculate the difference for the next iteration
            delta = max(delta, abs(val - V[state]))
        iterations+=1
        if delta < theta:
            break

    policy = np.zeros(env.observation_space.n, dtype=int)
    for state in range(env.observation_space.n):
        # Find the action that maximizes the expected value
        max_action_value = float('-inf')
        best_action = None
        for action in range(env.action_space.n):
            action_value = 0
            for probability, next_state, reward, _ in env.unwrapped.P[state][action]:
                if reward > 0:
                    action_value += probability * (reward + gamma * V[next_state])
                else:
                    action_value += probability * (reward + gamma * V[next_state])
            if action_value > max_action_value:
                max_action_value = action_value
                best_action = action
        policy[state] = best_action

    print(f"Optimal Policy: \n {policy.reshape(4,4)}")
    print(f"Value Function: \n {V.reshape(4,4)}")
    print(f"iterations:  \n {iterations}")

    # Render the environment with the optimal policy
    state = env.reset()[0]  # Starts at position 0.
    env.render()  # Renders the map.
    terminated = False  # Checks if the agent has fallen reached the goal.
    while not terminated:
        action = policy[state]  # Given the state, pass it into the policy function and return the correct action.
        state, reward, terminated, _, _ = env.step(
            action)  # Take the action and return the obs. state, reward, if terminated, and more.
        env.render()  # Render the new move.

    # Plot the policy and state value.
    plot_values_and_policy(V, policy, env)


# Used to run the Dynamic method.
if __name__ == '__main__':
    run(1)

# Remember, when rendering, the optimal policy is like a grid.
# For example, if [1 2 1 0 1 0 1 0 2 1 1 0 0 2 2 0] is returned, it is in the index order of a typical unit indexed array [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]. That is, if we pass in the state 1 into our policy function (policy[1]), output will be the best action, which in our case is 1 (down).
# Another example would be Policy[4], which returns 1 (down).