import gymnasium as gym
import numpy as np

def run(episodes):
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    
    epsilon = 0.1
    gamma = 0.9
    pi = np.ones((env.observation_space.n, env.action_space.n)) * epsilon / env.action_space.n
    Q = np.random.rand(env.observation_space.n, env.action_space.n) * 0.01
    returns = {(s, a): [] for s in range(env.observation_space.n) for a in range(env.action_space.n)}
    
    success_rate = 0
    avg_steps_to_goal = 0
    avg_reward_per_episode = 0
    optimal_paths = []  # Store all optimal paths

    for _ in range(episodes):
        episode = []
        state = env.reset()[0]
        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated:
            if np.random.default_rng().random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = new_state
            steps += 1

        avg_steps_to_goal += steps
        avg_reward_per_episode += reward

        if reward == 1:
            success_rate += 1
            optimal_paths.append(episode)

        G = 0
        for step in reversed(episode):
            state, action, reward = step
            G = (gamma * G) + reward
            if (state, action) not in [(x[0], x[1]) for x in episode[:-1]]:
                returns[(state, action)].append(G)
                Q[state, action] = np.mean(returns[((state, action))])
                optimalAction = np.argmax(Q[state, :])
                for a in range(env.action_space.n):
                    if a == optimalAction:
                        pi[state, a] = 1 - epsilon + (epsilon / env.action_space.n)
                    else:
                        pi[state, a] = epsilon / env.action_space.n
                        
    success_rate /= episodes
    avg_steps_to_goal /= episodes
    avg_reward_per_episode /= episodes

    print("Success Rate:", success_rate)
    print("Average Steps to Reach Goal:", avg_steps_to_goal)
    print("Average Reward per Episode:", avg_reward_per_episode)            
    
    print("Q-values:")    
    print(Q)
    print("Policy:")
    print(pi)
    print("Number of Optimal Paths Found:", len(optimal_paths))

    return Q, pi, optimal_paths

if __name__ == '__main__':
    Q, pi, optimal_paths = run(2500)

    # Select the optimal path with the highest average reward
    if optimal_paths:
        avg_rewards = [sum([step[2] for step in path]) / len(path) for path in optimal_paths]
        best_index = np.argmax(avg_rewards)
        best_path = optimal_paths[best_index]
        print(best_path)

        print("Rendering best optimal path:")
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human")
        state = env.reset()[0]
        terminated = False
        while not terminated:
            action = np.argmax(pi[state,:])
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state
            env.render()
