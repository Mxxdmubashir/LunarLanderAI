import gymnasium as gym
from stable_baselines3 import PPO


# Create the Lunar Lander environment
env = gym.make("LunarLander-v3")


print("🚀 Starting PPO training...")
print("The AI is learning how to land the rocket.")
print()


# Create the PPO brain
model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)


# Train the AI
model.learn(total_timesteps=100_000)


# Save the trained AI
model.save("lunar_lander_ppo")


# Close the environment
env.close()


print()
print("✅ Training complete!")
print("Model saved as: lunar_lander_ppo")