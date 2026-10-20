"""
Educational reward-hacking demonstration (toy example, not LunarLander training).

The designer wants an agent to reach the goal. The flawed reward gives points
for moving, regardless of direction. The agent can earn reward by moving away
from the goal. This isolates the reward-design idea without claiming it changes
the trained LunarLander policy.
"""
import random

START = 0
GOAL = 10
STEPS = 12

def run_episode(reward_mode, seed=7):
    random.seed(seed)
    position = START
    total_reward = 0
    trace = []
    for step in range(STEPS):
        # A toy policy exploiting the flawed proxy: move left to collect movement points.
        if reward_mode == "flawed":
            action = -1
        else:
            action = 1 if position < GOAL else 0

        old_distance = abs(GOAL - position)
        position += action
        new_distance = abs(GOAL - position)

        if reward_mode == "flawed":
            reward = 1 if action != 0 else 0  # Bad proxy: rewards any movement.
        else:
            reward = old_distance - new_distance  # Better proxy: reward progress toward goal.

        if position == GOAL:
            reward += 10
        total_reward += reward
        trace.append((step + 1, position, reward, total_reward))
        if position == GOAL:
            break
    return position, total_reward, trace

def main():
    for mode, title in [
        ("flawed", "FLAWED REWARD: any movement earns a point"),
        ("fixed", "IMPROVED REWARD: progress toward the goal is rewarded"),
    ]:
        position, total, trace = run_episode(mode)
        print("\\n" + title)
        print("-" * len(title))
        print("step | position | reward | cumulative reward")
        for step, pos, reward, cumulative in trace:
            print(f"{step:>4} | {pos:>8} | {reward:>6} | {cumulative:>17}")
        print(f"Final position: {position}; goal: {GOAL}; total reward: {total}")
        if position != GOAL:
            print("RESULT: high proxy reward may not mean the real goal was achieved.")
        else:
            print("RESULT: the agent reached the intended goal.")

if __name__ == "__main__":
    main()
