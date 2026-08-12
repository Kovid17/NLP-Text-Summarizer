"""
notebooks/01_EDA_CNN_SAMSum.py
Phase 1 — Exploratory Data Analysis: CNN/DailyMail + SAMSum
Run as script: python notebooks/01_EDA_CNN_SAMSum.py
Or open as Jupyter notebook (rename to .ipynb cells manually)
"""
# %% [markdown]
# # Phase 1: EDA — CNN/DailyMail & SAMSum
# **Goal**: Understand dataset statistics, token length distributions,
# and vocabulary before any training.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

from src.dataset_loader import DatasetLoader
from src.utils import set_seed

set_seed(42)
sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({"figure.facecolor": "#1e1e2e", "axes.facecolor": "#1e1e2e",
                     "text.color": "white", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white"})


# %% [markdown]
# ## 1. Load Datasets
# %%
print("Loading CNN/DailyMail ...")
cnn_loader = DatasetLoader("cnn_dailymail")
cnn_splits = cnn_loader.load()
cnn_train  = cnn_splits["train"].to_pandas()

print("\nLoading SAMSum ...")
sam_loader = DatasetLoader("samsum")
sam_splits = sam_loader.load()
sam_train  = sam_splits["train"].to_pandas()

print(f"\nCNN/DM  train: {len(cnn_train):,} rows")
print(f"SAMSum  train: {len(sam_train):,} rows")


# %% [markdown]
# ## 2. Sample Inspection
# %%
print("\n─── CNN/DailyMail Sample ───────────────────────────")
print("INPUT :", cnn_train["input_text"].iloc[0][:400], "...")
print("TARGET:", cnn_train["target_text"].iloc[0])

print("\n─── SAMSum Sample ──────────────────────────────────")
print("INPUT :", sam_train["input_text"].iloc[0][:400], "...")
print("TARGET:", sam_train["target_text"].iloc[0])


# %% [markdown]
# ## 3. Token / Word Length Distributions
# %%
def word_lengths(df):
    return (df["input_text"].str.split().str.len(),
            df["target_text"].str.split().str.len())

cnn_inp_len, cnn_tgt_len = word_lengths(cnn_train)
sam_inp_len, sam_tgt_len = word_lengths(sam_train)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Word Length Distributions — Phase 1 Datasets", fontsize=15, fontweight="bold")

axes[0, 0].hist(cnn_inp_len.clip(0, 1200), bins=50, color="#a78bfa", edgecolor="none")
axes[0, 0].set_title("CNN/DM — Input Length")
axes[0, 0].set_xlabel("Words")

axes[0, 1].hist(cnn_tgt_len.clip(0, 200), bins=40, color="#60a5fa", edgecolor="none")
axes[0, 1].set_title("CNN/DM — Summary Length")
axes[0, 1].set_xlabel("Words")

axes[1, 0].hist(sam_inp_len.clip(0, 600), bins=50, color="#34d399", edgecolor="none")
axes[1, 0].set_title("SAMSum — Dialogue Length")
axes[1, 0].set_xlabel("Words")

axes[1, 1].hist(sam_tgt_len.clip(0, 80), bins=40, color="#fbbf24", edgecolor="none")
axes[1, 1].set_title("SAMSum — Summary Length")
axes[1, 1].set_xlabel("Words")

plt.tight_layout()
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/phase1_length_distributions.png", dpi=150, bbox_inches="tight")
print("Saved: reports/figures/phase1_length_distributions.png")
plt.show()


# %% [markdown]
# ## 4. Summary Statistics Table
# %%
stats = pd.DataFrame({
    "Dataset":          ["CNN/DailyMail", "CNN/DailyMail", "SAMSum", "SAMSum"],
    "Column":           ["Input",         "Target",        "Input",  "Target"],
    "Mean Words":       [cnn_inp_len.mean(), cnn_tgt_len.mean(),
                         sam_inp_len.mean(), sam_tgt_len.mean()],
    "Median Words":     [cnn_inp_len.median(), cnn_tgt_len.median(),
                         sam_inp_len.median(), sam_tgt_len.median()],
    "Max Words":        [cnn_inp_len.max(), cnn_tgt_len.max(),
                         sam_inp_len.max(), sam_tgt_len.max()],
    "Compression Ratio": [
        cnn_inp_len.mean() / cnn_tgt_len.mean(),
        "-",
        sam_inp_len.mean() / sam_tgt_len.mean(),
        "-",
    ],
})
print("\n", stats.to_string(index=False))


# %% [markdown]
# ## 5. Top Vocabulary Words
# %%
def top_words(series, n=20):
    all_words = " ".join(series.dropna()).lower().split()
    return Counter(all_words).most_common(n)

print("\n── CNN/DM Top 20 Input Words ──")
print(top_words(cnn_train["input_text"]))

print("\n── SAMSum Top 20 Dialogue Words ──")
print(top_words(sam_train["input_text"]))


# %% [markdown]
# ## 6. Compression Ratio Distribution
# %%
cnn_ratio = cnn_inp_len / cnn_tgt_len.clip(1)
sam_ratio = sam_inp_len / sam_tgt_len.clip(1)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Compression Ratio Distribution (Input Words / Summary Words)", fontsize=13, fontweight="bold")

ax[0].hist(cnn_ratio.clip(0, 30), bins=40, color="#a78bfa", edgecolor="none")
ax[0].set_title("CNN/DailyMail")
ax[0].set_xlabel("Compression Ratio")
ax[0].axvline(cnn_ratio.median(), color="#fbbf24", linestyle="--", label=f"Median={cnn_ratio.median():.1f}x")
ax[0].legend()

ax[1].hist(sam_ratio.clip(0, 30), bins=40, color="#34d399", edgecolor="none")
ax[1].set_title("SAMSum")
ax[1].set_xlabel("Compression Ratio")
ax[1].axvline(sam_ratio.median(), color="#fbbf24", linestyle="--", label=f"Median={sam_ratio.median():.1f}x")
ax[1].legend()

plt.tight_layout()
plt.savefig("reports/figures/phase1_compression_ratio.png", dpi=150, bbox_inches="tight")
print("Saved: reports/figures/phase1_compression_ratio.png")
plt.show()

print("\n✅ Phase 1 EDA complete!")
