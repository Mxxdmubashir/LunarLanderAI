import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Load the trained model
model = PPO.load("lunar_lander_ppo")

# Create the environment without graphics
env = gym.make("LunarLander-v3")

# Evaluation settings
num_episodes = 100
rewards = []
successful_landings = 0

print("Evaluating trained Lunar Lander AI...")
print(f"Running {num_episodes} episodes...\n")

for episode in range(num_episodes):
    observation, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

    rewards.append(total_reward)

    # A successful landing usually ends with both legs
    # in contact with the ground, without a crash.
    left_leg = observation[6] > 0.5
    right_leg = observation[7] > 0.5

    if left_leg and right_leg:
        successful_landings += 1

    print(
        f"Episode {episode + 1:3d}: "
        f"Reward = {total_reward:8.2f}"
    )

env.close()

mean_reward = np.mean(rewards)
std_reward = np.std(rewards)

print("\n========== EVALUATION RESULTS ==========")
print(f"Mean reward: {mean_reward:.2f}")
print(f"Reward standard deviation: {std_reward:.2f}")
print(f"Both legs touching ground at episode end: "
      f"{successful_landings}/{num_episodes}")

if mean_reward >= 200:
    print("The model meets the usual mean-reward benchmark!")
else:
    print("More training may improve the model.")

print("========================================")