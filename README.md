# 🚀 LunarLanderAI — Reinforcement Learning with PPO

An AI agent trained using Reinforcement Learning to control a lunar lander,
manage its movements, and attempt safe landings in a simulated environment.

## 🌌 Project Overview

LunarLanderAI explores how an autonomous agent can learn to make decisions
through trial and error using rewards and penalties.

The project uses Proximal Policy Optimization (PPO) to train an agent
in the Gymnasium LunarLander environment.

## 🎯 Objectives

- Train an autonomous Lunar Lander agent.
- Apply reinforcement learning using PPO.
- Evaluate model performance across multiple episodes.
- Visualize and compare evaluation results.
- Explore reward design and reward hacking.

## 🛠️ Technologies Used

- Python
- Reinforcement Learning
- Proximal Policy Optimization (PPO)
- Gymnasium
- Stable-Baselines3
- Box2D
- Matplotlib
- Pandas
- Git and GitHub

## 🧠 How It Works

1. The agent observes the environment.
2. It selects actions to control the lander.
3. The environment provides rewards or penalties.
4. PPO updates the agent's policy during training.
5. The trained model is evaluated across multiple episodes.

## 📊 Performance Evaluation

The improved model was evaluated over 100 episodes.

| Metric | Recorded Result |
|---|---:|
| Mean episode reward | 258.74 |
| Reward standard deviation | 46.34 |
| Episodes meeting the landing-state heuristic | 84/100 |

**Note:** The landing-state heuristic checks the lander's final state.
It is not a verified successful-landing rate. Further evaluation is needed
to distinguish all safe landings from crashes.

## 📈 Performance Graphs

Evaluation graphs are available in the `graphs/` directory.

- `mean_reward_comparison.png` — compares model rewards.
- `landing_indicators.png` — visualizes landing-state indicators.
- `reward_variability.png` — compares reward variability.

## 📁 Project Structure

```text
LunarLanderAI/
├── graphs/
├── reports/
├── scripts/
├── checkpoints/
├── .gitignore
├── README.md
└── requirements.txt