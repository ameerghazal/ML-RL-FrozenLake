import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def value_iteration(env, gamma=0.9, epsilon=1e-10):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    V = np.zeros(num_states)  # Initialize value function

    while True:
        delta = 0
        for state in range(num_states):
            v = V[state]
            V[state] = max([sum([p*(r + gamma*V[next_state]) for p, next_state, r, _ in env.unwrapped.P[state][action]]) 
                           for action in range(num_actions)])
            delta = max(delta, abs(v - V[state]))

        if delta < epsilon:
            break

    # Extract optimal policy from the optimal value function
    policy = np.zeros(num_states, dtype=int)
    for state in range(num_states):
        policy[state] = np.argmax([sum([p*(r + gamma*V[next_state]) for p, next_state, r, _ in env.unwrapped.P[state][action]])
                                   for action in range(num_actions)])

    return policy

def policy_iteration(env, gamma=0.9, epsilon=1e-10):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy = np.ones(num_states, dtype=int)  # Initialize policy
    V = np.zeros(num_states)  # Initialize value function

    while True:
        # Policy Evaluation
        while True:
            delta = 0
            for state in range(num_states):
                v = V[state]
                action = policy[state]
                V[state] = sum([p*(r + gamma*V[next_state]) for p, next_state, r, _ in env.P[state][action]])
                delta = max(delta, abs(v - V[state]))

            if delta < epsilon:
                break

        # Policy Improvement
        policy_stable = True
        for state in range(num_states):
            old_action = policy[state]
            policy[state] = np.argmax([sum([p*(r + gamma*V[next_state]) for p, next_state, r, _ in env.P[state][action]])
                                       for action in range(num_actions)])
            if old_action != policy[state]:
                policy_stable = False

        if policy_stable:
            break

    return policy


# Example Usage:
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode=None)
optimal_policy_vi = value_iteration(env)
optimal_policy_pi = policy_iteration(env)
print("Optimal Policy (Value Iteration):", optimal_policy_vi)
print("Optimal Policy (Policy Iteration):", optimal_policy_pi)