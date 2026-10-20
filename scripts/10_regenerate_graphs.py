"""Regenerate project charts from the recorded evaluation results."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "graphs" / "evaluation_results.csv"
OUT = ROOT / "graphs"
df = pd.read_csv(DATA)

# Keep the labels and caveats in the output so charts are not misrepresented.
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(df["model/run"], df["mean_reward"])
ax.axhline(0, linewidth=1)
ax.set_title("LunarLander-v3 Mean Episode Reward")
ax.set_ylabel("Mean reward (100 episodes)")
ax.tick_params(axis="x", labelrotation=12)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "mean_reward_regenerated.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved graphs/mean_reward_regenerated.png")
