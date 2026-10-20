import gymnasium as gym
from stable_baselines3 import PPO
import time

# Load the trained AI model
model = PPO.load("lunar_lander_ppo")

# Create the game window
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

print("🚀 Trained Lunar Lander AI")
print("Watch the AI control the rocket!")
print()

# Run 5 test episodes
for episode in range(1, 6):

    observation, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # AI chooses an action
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        # Apply the action
        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

        time.sleep(0.01)

    print(
        f"Episode {episode}: "
        f"Total Reward = {total_reward:.2f}"
    )

    time.sleep(1)

env.close()

print("\n✅ AI testing completed!")