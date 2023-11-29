import gymnasium as gym
import numpy as np

#2.2
def estimate_functions(env, episodes):
    transition_counts = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    reward_sums = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    state_action_count = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = env.action_space.sample()  # Random action
            next_state, reward, terminated, truncated, _ = env.step(action)
            transition_counts[state, action, next_state] += 1
            reward_sums[state, action, next_state] += reward
            state_action_count[state, action] += 1

            state = next_state

    transition_p = np.divide(transition_counts, state_action_count[:, :, None], where=state_action_count[:, :, None] != 0)
    reward_estimates = np.divide(reward_sums, state_action_count[:, :, None], where=state_action_count[:, :, None] != 0)

    return transition_p, reward_estimates

#2.3
def value_iteration(transition_p, reward_estimates, gamma=0.95, epsilon=1e-20):
    num_states, num_actions, _ = reward_estimates.shape
    V = np.zeros(num_states)
    
    while True:
        prev_V = np.copy(V)
        temp = np.zeros((num_states, num_actions))
        
        for s in range(num_states):
            for a in range(num_actions):
                temp[s, a] = np.sum(transition_p[s, a] * (reward_estimates[s, a] + gamma * prev_V))
        
        V = np.max(temp, axis=1)
        
        if np.max(np.abs(prev_V - V)) < epsilon:
            break
    return V

#2.4
def extract_policy(transition_p, reward_estimates, optimal_values, gamma=0.95):
    n_states, n_actions, _ = reward_estimates.shape
    policy = np.zeros(n_states, dtype=int)
    
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            action_values[a] = np.sum(transition_p[s, a] * (reward_estimates[s, a] + gamma * optimal_values))
        policy[s] = np.argmax(action_values)
    
    return policy

#2.5
def act_optimally(env, policy):
    state, _ = env.reset()
    terminated = False
    truncated = False

    while not terminated or not truncated:
        action = policy[state]
        next_state, _, terminated, truncated, _ = env.step(action)
        state = next_state

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4",  is_slippery=True, ) #initialization
transition_p, reward_estimates = estimate_functions(env, episodes=1000)
optimal_values = value_iteration(transition_p, reward_estimates)
optimal_policy = extract_policy(transition_p, reward_estimates, optimal_values)
act_optimally(env, optimal_policy)
