import gymnasium as gym
env = gym.make("Blackjack-v1", natural=False, sab=False) # Initializing environments
observation, info = env.reset()

