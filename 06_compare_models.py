import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

MODEL_FILES = {
    "Original AI": "lunar_lander_ppo.zip",
    "Improved AI": "checkpoints/lander_1000352_steps.zip",
}
NUM_EPISODES = 100


def evaluate_model(model_path, env):
    model = PPO.load(model_path)

    rewards = []
    successful_landings = 0

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

        rewards.append(total_reward)

        # Check both legs and whether the lander is
        # upright and moving slowly at episode end.
        legs_touching = obs[6] > 0.5 and obs[7] > 0.5
        nearly_upright = abs(obs[4]) < 0.2
        moving_slowly = abs(obs[2]) < 0.2 and abs(obs[3]) < 0.2

        if legs_touching and nearly_upright and moving_slowly:
            successful_landings += 1

    return (
        np.mean(rewards),
        np.std(rewards),
        successful_landings
    )


env = gym.make("LunarLander-v3")

for name, path in MODEL_FILES.items():
    mean_reward, std_reward, landings = evaluate_model(
        path,
        env
    )

    print(f"\n{name}")
    print("-" * 35)
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Standard deviation: {std_reward:.2f}")
    print(f"Landing indicators: {landings}/{NUM_EPISODES}")

env.close()