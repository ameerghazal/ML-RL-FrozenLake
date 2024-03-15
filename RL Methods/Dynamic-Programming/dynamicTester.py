import numpy as np
import gym

def value_iteration(env, gamma=0.9, epsilon=1e-6):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize value function
    V = np.zeros(num_states)

    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]

            # Calculate Q-values for all actions in state s
            Q_values = np.zeros(num_actions)
            for a in range(num_actions):
                for transition in env.P[s][a]:
                    next_state, prob, reward, _ = transition
                    next_state = int(next_state)  # Convert to integer
                    Q_values[a] += prob * (reward + gamma * V[next_state])

            # Update value function
            V[s] = np.max(Q_values)

            delta = max(delta, abs(v - V[s]))

        # Check for convergence
        if delta < epsilon:
            break

    # Extract optimal policy from value function
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        Q_values = np.zeros(num_actions)
        for a in range(num_actions):
            for transition in env.P[s][a]:
                next_state, prob, reward, _ = transition
                next_state = int(next_state)  # Convert to integer
                Q_values[a] += prob * (reward + gamma * V[next_state])

        policy[s] = np.argmax(Q_values)

    return policy

def main():
    # Create FrozenLake environment with "is_slippery" set to False
    env = gym.make("FrozenLake-v1", is_slippery=False)

    # Run value iteration to find optimal policy
    optimal_policy = value_iteration(env)

    # Print the optimal policy
    print("Optimal Policy:")
    print(optimal_policy)

if __name__ == "__main__":
    main()
