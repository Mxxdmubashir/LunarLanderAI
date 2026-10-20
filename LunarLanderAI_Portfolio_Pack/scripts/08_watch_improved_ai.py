"""Watch the trained LunarLander PPO agent in a visible window."""
import gymnasium as gym
from stable_baselines3 import PPO

MODEL_PATH = "checkpoints/lander_1000352_steps.zip"
EPISODES_TO_WATCH = 5

def main():
    model = PPO.load(MODEL_PATH)
    env = gym.make("LunarLander-v3", render_mode="human")
    try:
        for episode in range(1, EPISODES_TO_WATCH + 1):
            obs, info = env.reset()
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
            print(f"Episode {episode}: reward={total_reward:.2f}; "
                  f"terminated={terminated}; truncated={truncated}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
