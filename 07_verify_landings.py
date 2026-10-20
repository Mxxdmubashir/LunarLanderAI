import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Your trained AI model
MODEL_PATH = "checkpoints/lander_1000352_steps.zip"

# Number of test episodes
NUM_EPISODES = 100

# Load the model
model = PPO.load(MODEL_PATH)

# Create the Lunar Lander environment
env = gym.make("LunarLander-v3")

rewards = []
successful_landings = 0
crashes = 0
other_endings = 0

print("\nVERIFYING IMPROVED LUNAR LANDER AI")
print("=" * 45)

for episode in range(NUM_EPISODES):
    obs, info = env.reset()

    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

    rewards.append(total_reward)

    # LunarLander observation:
    # obs[0] = horizontal position
    # obs[1] = vertical position
    # obs[2] = horizontal velocity
    # obs[3] = vertical velocity
    # obs[4] = angle
    # obs[5] = angular velocity
    # obs[6] = left leg contact
    # obs[7] = right leg contact

    left_leg = obs[6] > 0.5
    right_leg = obs[7] > 0.5

    upright = abs(obs[4]) < 0.2
    slow_horizontal = abs(obs[2]) < 0.2
    slow_vertical = abs(obs[3]) < 0.2

    # Strict landing-state check
    landing_state = (
        left_leg
        and right_leg
        and upright
        and slow_horizontal
        and slow_vertical
    )

    if terminated and not truncated and landing_state:
        successful_landings += 1

    elif terminated and not truncated:
        crashes += 1

    else:
        other_endings += 1

    print(
        f"Episode {episode + 1:3d}/{NUM_EPISODES} | "
        f"Reward: {total_reward:7.2f} | "
        f"Landing state: {'YES' if landing_state else 'NO'}"
    )

env.close()

print("\n" + "=" * 45)
print("FINAL VERIFICATION RESULTS")
print("=" * 45)

print(f"Mean reward: {np.mean(rewards):.2f}")
print(f"Reward standard deviation: {np.std(rewards):.2f}")

print(
    f"Strict landing-state episodes: "
    f"{successful_landings}/{NUM_EPISODES}"
)

print(
    f"Strict landing-state rate: "
    f"{successful_landings / NUM_EPISODES * 100:.1f}%"
)

print(f"Terminated without passing the landing check: {crashes}")
print(f"Truncated episodes: {other_endings}")

print("\nVerification completed!")