import gymnasium as gym
import numpy as np
def estimate_functions(env, episodes):
    transition_counts = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    reward_sums = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
    state_action_count = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(episodes):
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

    transition_probs = np.divide(transition_counts, state_action_count[:, :, None], where=state_action_count[:, :, None] != 0)
    reward_estimates = np.divide(reward_sums, state_action_count[:, :, None], where=state_action_count[:, :, None] != 0)

    return transition_probs, reward_estimates


def value_iteration(transition_probs, reward_estimates, gamma=0.95, epsilon=1e-20):
    num_states, num_actions, _ = reward_estimates.shape
    V = np.zeros(num_states)
    
    while True:
        prev_V = np.copy(V)
        temp_values = np.zeros((num_states, num_actions))
        
        for s in range(num_states):
            for a in range(num_actions):
                temp_values[s, a] = np.sum(transition_probs[s, a] * (reward_estimates[s, a] + gamma * prev_V))
        
        V = np.max(temp_values, axis=1)
        
        if np.max(np.abs(prev_V - V)) < epsilon:
            break
    return V

def extract_policy(transition_probs, reward_estimates, optimal_value_function, gamma=0.99):
    num_states, num_actions, _ = reward_estimates.shape
    policy = np.zeros(num_states, dtype=int)
    
    for s in range(num_states):
        action_values = np.zeros(num_actions)
        for a in range(num_actions):
            action_values[a] = np.sum(transition_probs[s, a] * (reward_estimates[s, a] + gamma * optimal_value_function))
        policy[s] = np.argmax(action_values)
    
    return policy



# Initialize FrozenLake environment
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4",  is_slippery=True, ) #initialization
env.metadata['render_fps'] = 500
# Estimate transition and reward functions based on 1000 episodes
transition_probs, reward_estimates = estimate_functions(env, episodes=1000)
optimal_value_function = value_iteration(transition_probs, reward_estimates)
optimal_policy = extract_policy(transition_probs, reward_estimates, optimal_value_function)
print(optimal_policy)
