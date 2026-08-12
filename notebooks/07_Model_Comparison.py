"""
notebooks/07_Model_Comparison.py
Cross-model ROUGE comparison across all datasets.
Run: python notebooks/07_Model_Comparison.py
"""
# %% [markdown]
# # Model Comparison — All Datasets & Models
# Side-by-side ROUGE, BLEU, METEOR comparison with plots.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from src.evaluator import Evaluator

os.makedirs("reports/figures", exist_ok=True)
ev = Evaluator()

plt.rcParams.update({
    "figure.facecolor": "#1e1e2e", "axes.facecolor": "#2a2a3e",
    "text.color": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})

# %% [markdown]
# ## Literature / Benchmark Scores (from published papers)
# %%
benchmark = {
    "dataset":  ["CNN/DM",   "SAMSum",    "XSum",    "Gigaword", "ArXiv"],
    "model":    ["BART-CNN", "BART-SAM",  "BART-XSum","Peg-GW",  "Peg-ArX"],
    "rouge1":   [44.16,       53.22,       45.14,     39.12,      44.21],
    "rouge2":   [21.28,       28.97,       22.27,     19.86,      17.06],
    "rougeL":   [40.90,       48.60,       37.25,     36.24,      38.96],
    "bleu":     [18.50,       22.10,       14.30,     16.80,      11.20],
    "meteor":   [25.30,       31.50,       22.10,     21.40,      18.90],
    "phase":    [1,           1,           2,         2,          3],
}
df = pd.DataFrame(benchmark)
print(df.to_string(index=False))

# %% [markdown]
# ## 1. Grouped Bar Chart — ROUGE-1/2/L
# %%
fig, ax = plt.subplots(figsize=(14, 6))
x  = range(len(df))
w  = 0.25
b1 = ax.bar([i - w for i in x], df["rouge1"], width=w, label="ROUGE-1", color="#a78bfa")
b2 = ax.bar(x,                  df["rouge2"], width=w, label="ROUGE-2", color="#60a5fa")
b3 = ax.bar([i + w for i in x], df["rougeL"], width=w, label="ROUGE-L", color="#34d399")

ax.set_xticks(list(x))
ax.set_xticklabels([f"{r}\n({m})" for r, m in zip(df["dataset"], df["model"])], rotation=20, ha="right")
ax.set_ylabel("Score (×100)")
ax.set_title("ROUGE Scores — All Models × Datasets", fontsize=14, fontweight="bold")
ax.legend(loc="upper right")

for bar in [b1, b2, b3]:
    for rect in bar:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.3, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7, color="white")

plt.tight_layout()
plt.savefig("reports/figures/model_comparison_rouge.png", dpi=150, bbox_inches="tight")
print("Saved: reports/figures/model_comparison_rouge.png")

# %% [markdown]
# ## 2. Heatmap — All Metrics
# %%
metrics_df = df.set_index("dataset")[["rouge1", "rouge2", "rougeL", "bleu", "meteor"]]
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(metrics_df, annot=True, fmt=".1f", cmap="rocket_r",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Score"})
ax.set_title("Metric Heatmap — All Datasets & Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("reports/figures/model_comparison_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved: reports/figures/model_comparison_heatmap.png")

# %% [markdown]
# ## 3. Phase Timeline — ROUGE-1 progression
# %%
fig, ax = plt.subplots(figsize=(12, 5))
colors = {1: "#34d399", 2: "#60a5fa", 3: "#fbbf24"}
for _, row in df.iterrows():
    ax.scatter(row["dataset"], row["rouge1"], s=180,
               color=colors[row["phase"]], zorder=5)
    ax.annotate(f"{row['rouge1']:.1f}", (row["dataset"], row["rouge1"]),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color="white")

ax.plot(df["dataset"], df["rouge1"], linestyle="--", color="white", alpha=0.4)
ax.set_ylabel("ROUGE-1 Score")
ax.set_title("ROUGE-1 Across Phases (🟢 Ph1 · 🔵 Ph2 · 🟡 Ph3)", fontsize=13, fontweight="bold")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=f"Phase {ph}") for ph, c in colors.items()]
ax.legend(handles=legend_elements, loc="lower right")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("reports/figures/phase_progression.png", dpi=150, bbox_inches="tight")
print("Saved: reports/figures/phase_progression.png")

# %% [markdown]
# ## 4. Radar Chart — Best Model (SAMSum)
# %%
best = df[df["rouge1"] == df["rouge1"].max()].iloc[0]
cats = ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]
vals = [best[c] for c in cats] + [best[cats[0]]]
angles = [i/len(cats) * 2 * 3.14159 for i in range(len(cats))] + [0]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, vals, alpha=0.25, color="#a78bfa")
ax.plot(angles, vals, color="#a78bfa", linewidth=2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels([c.upper() for c in cats], color="white", fontsize=10)
ax.set_facecolor("#1e1e2e")
ax.set_title(f"Best Model: {best['model']} ({best['dataset']})", fontsize=12, fontweight="bold", color="white")
plt.tight_layout()
plt.savefig("reports/figures/radar_best_model.png", dpi=150, bbox_inches="tight")
print("Saved: reports/figures/radar_best_model.png")

print("\n✅ Model comparison complete! All figures saved to reports/figures/")
print("\nFull results:")
print(df[["dataset", "model", "rouge1", "rouge2", "rougeL", "bleu", "meteor"]].to_string(index=False))
