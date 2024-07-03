import os
import numpy as np
import torch as th
from stable_baselines3 import PPO,DQN
from env_test import air_env
import matplotlib.pyplot as plt
def evaluate_model(model, env, n_eval_episodes=10):
    episode_rewards = []

    for episode in range(n_eval_episodes):
        episode_reward = 0
        obs,_ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, reward, done, _,info = env.step(action)
            episode_reward += reward


        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

model = DQN.load("tmp/best_model/best_model_DQN.zip")
env = air_env()
print(evaluate_model(model=model, env=env, n_eval_episodes=100))