import gymnasium as gym
import time

# Create the Lunar Lander game
env = gym.make("LunarLander-v3", render_mode="human")

# Start a new episode
observation, info = env.reset()

print("🚀 Lunar Lander AI Simulation Started!")
print("Observation values:", observation)

try:
    for step in range(1000):

        # Choose a random action
        action = env.action_space.sample()

        # Perform the action
        observation, reward, terminated, truncated, info = env.step(action)

        # Display the results
        print(
            f"Step: {step + 1} | "
            f"Action: {action} | "
            f"Reward: {reward:.2f}"
        )

        # Stop if the rocket lands or crashes
        if terminated or truncated:
            print("\nEpisode finished!")
            print("Total steps:", step + 1)
            break

        time.sleep(0.02)

finally:
    env.close()
    print("Simulation closed.")