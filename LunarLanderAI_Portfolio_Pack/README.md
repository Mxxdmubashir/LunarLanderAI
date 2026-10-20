# Lunar Lander AI — PPO Reinforcement Learning

A beginner-friendly reinforcement-learning project using Gymnasium's `LunarLander-v3` environment and Stable-Baselines3 PPO. The agent learns a control policy through trial and error and is evaluated on held-out episodes.

## Current results

The following numbers were reported from local runs of this project. Each run evaluated 100 episodes.

| Run | Mean reward | Reward standard deviation | Episodes passing a landing-state check |
|---|---:|---:|---:|
| Original AI | -2.63 | 37.92 | 8/100 |
| Improved AI — comparison run | 258.74 | 46.34 | 85/100 |
| Improved AI — verification run | 262.06 | 55.49 | 84/100 |

**Interpretation:** the improved model's mean reward is much higher than the original model's. The landing counts are *heuristic final-state indicators*, not definitive success labels. The original and improved comparison checks were not perfectly identical, and the verification script does not have access to a dedicated success flag. Report these values accurately as indicators rather than claiming a confirmed 84% success rate.

## Model files expected in the working project

```text
LunarLanderAI/
├── lunar_lander_ppo.zip
├── checkpoints/
│   └── lander_1000352_steps.zip
├── scripts/
│   ├── 08_watch_improved_ai.py
│   ├── 09_reward_hacking_demo.py
│   └── 10_regenerate_graphs.py
├── graphs/
└── reports/
```

The model files are not included in this portfolio pack. Keep your trained model files in the project directory as shown above.

## Run the project

Activate your existing virtual environment in PowerShell:

```powershell
cd D:\LunarLanderAI
.\.venv\Scripts\Activate.ps1
```

### Watch the trained agent

Copy `scripts/08_watch_improved_ai.py` into your project and run:

```powershell
python scripts/08_watch_improved_ai.py
```

A Lunar Lander window should open and show five episodes. Close the window or let the episodes finish. This requires a working graphical desktop.

### Demonstrate reward hacking

```powershell
python scripts/09_reward_hacking_demo.py
```

This is a small, separate toy example, not a modified LunarLander training run. It shows why rewarding a proxy (any movement) can produce behavior that fails the real goal, while rewarding progress toward the goal is better aligned.

### Regenerate the mean-reward graph

Install dependencies if needed:

```powershell
python -m pip install matplotlib pandas
```

Then run:

```powershell
python scripts/10_regenerate_graphs.py
```

The included charts were generated from the results above. They are historical evaluation results, not live measurements.

## Reproducibility and limitations

- Environment: `LunarLander-v3` from Gymnasium.
- Algorithm: Proximal Policy Optimization (PPO), using Stable-Baselines3.
- Improved checkpoint: `checkpoints/lander_1000352_steps.zip`.
- Evaluation is stochastic across episode seeds, so means vary between runs.
- A strong mean reward is encouraging but does not alone prove a high success rate.
- The provided landing-state check uses leg contact, angle, and velocity thresholds; it is only a heuristic. A robust evaluation should also inspect termination behavior and, ideally, use a verified success criterion.
- The reward-hacking demo is deliberately a toy environment and should not be described as proof that the LunarLander policy itself reward-hacks.

## Suggested portfolio summary

Built and evaluated a PPO reinforcement-learning agent for Gymnasium LunarLander-v3 using Stable-Baselines3. Continued training from an initial checkpoint for approximately one million additional timesteps. Across 100 evaluation episodes, the improved checkpoint achieved a reported mean reward of 258.74 in one comparison run and 262.06 in a subsequent verification run, compared with -2.63 for the original model. Added visualization, result charts, and a separate educational demonstration of reward hacking. Landing figures are reported as heuristic state indicators rather than verified success labels.
