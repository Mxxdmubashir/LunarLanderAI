import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Create the environment
env = gym.make("LunarLander-v3")

# Load the existing trained model
model = PPO.load(
    "lunar_lander_ppo",
    env=env
)

# Save checkpoints every 100,000 additional steps
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="lander"
)

print("🚀 Continuing PPO training...")
print("Additional training: 1,000,000 timesteps")
print("Starting from your existing model.")
print()

# Continue training
model.learn(
    total_timesteps=1_000_000,
    callback=checkpoint_callback,
    reset_num_timesteps=False
)

# Save the improved model separately
model.save("lunar_lander_ppo_improved")

env.close()

print("\n✅ Additional training complete!")
print("Saved: lunar_lander_ppo_improved")