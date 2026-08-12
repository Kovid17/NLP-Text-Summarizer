"""
notebooks/04_Training_Phase1.py
Phase 1 — Training Walkthrough: CNN/DailyMail + SAMSum
Demonstrates the full training pipeline with ROUGE tracking and visualisation.
Run: python notebooks/04_Training_Phase1.py
"""
# %% [markdown]
# # Phase 1: Training — CNN/DailyMail & SAMSum
# Pipeline: Load → Tokenize → Train → Evaluate → Visualise

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("Agg")

from src.utils          import set_seed, get_device, get_model_config, get_run_dir
from src.dataset_loader import DatasetLoader
from src.preprocessor   import Preprocessor
from src.model_factory  import ModelFactory
from src.trainer        import Trainer
from src.inference      import SummarizationInference
from src.evaluator      import Evaluator

set_seed(42)
device = get_device()
print(f"Device: {device}")

# %% [markdown]
# ## Step 1 — Load CNN/DailyMail (subset for demo)
# %%
print("\n[1/6] Loading CNN/DailyMail dataset ...")
cnn_loader = DatasetLoader("cnn_dailymail")
splits     = cnn_loader.load()

train_ds = splits["train"]
val_ds   = splits["validation"]
test_ds  = splits["test"]
print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

# %% [markdown]
# ## Step 2 — Load Model & Tokenizer
# %%
print("\n[2/6] Loading BART-Large-CNN ...")
cfg              = get_model_config("cnn_dailymail")
model, tokenizer = ModelFactory.from_dataset("cnn_dailymail", device=device)
print(f"  Model loaded: {cfg['model_name']}")

# %% [markdown]
# ## Step 3 — Tokenize & Create DataLoaders
# %%
print("\n[3/6] Tokenizing & building DataLoaders ...")
pre          = Preprocessor(tokenizer, cfg)
train_loader = pre.get_dataloader(train_ds, shuffle=True)
val_loader   = pre.get_dataloader(val_ds,   shuffle=False)
print(f"  Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

# %% [markdown]
# ## Step 4 — Train
# %%
print("\n[4/6] Training ...")
run_dir = get_run_dir("cnn_dailymail")
trainer = Trainer(model, tokenizer, cfg, device, output_dir=str(run_dir))
history = trainer.train(train_loader, val_loader, val_ds)

# %% [markdown]
# ## Step 5 — Visualise Training History
# %%
print("\n[5/6] Plotting training history ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Phase 1 Training — CNN/DailyMail (BART-Large-CNN)", fontsize=14, fontweight="bold")
fig.patch.set_facecolor("#1e1e2e")

for ax in axes:
    ax.set_facecolor("#1e1e2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

epochs_list = list(range(1, len(history["train_loss"]) + 1))

axes[0].plot(epochs_list, history["train_loss"], marker="o", color="#a78bfa", linewidth=2, label="Train Loss")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(epochs_list, history["val_rouge1"], marker="o", color="#60a5fa", linewidth=2, label="ROUGE-1")
axes[1].plot(epochs_list, history["val_rouge2"], marker="s", color="#34d399", linewidth=2, label="ROUGE-2")
axes[1].plot(epochs_list, history["val_rougeL"], marker="^", color="#fbbf24", linewidth=2, label="ROUGE-L")
axes[1].set_title("Validation ROUGE Scores")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].legend()

plt.tight_layout()
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/phase1_training_history.png", dpi=150, bbox_inches="tight")
print("  Saved: reports/figures/phase1_training_history.png")

# %% [markdown]
# ## Step 6 — Inference & Final Evaluation
# %%
print("\n[6/6] Final evaluation on test set ...")
inferencer  = SummarizationInference(model, tokenizer, cfg, device)
ev          = Evaluator()

# Test on first 200 samples for speed
test_sub    = test_ds.select(range(min(200, len(test_ds))))
predictions = inferencer.batch_summarize(test_sub["input_text"], batch_size=4)
references  = test_sub["target_text"]
scores      = ev.compute(predictions, references)

ev.print_scores(scores, dataset="cnn_dailymail", model="BART-Large-CNN")
ev.save_csv(scores, "cnn_dailymail", "BART-Large-CNN")

# Show 3 qualitative examples
print("\n── Qualitative Examples ─────────────────────────────────────")
for i in range(3):
    print(f"\n[{i+1}] INPUT   : {test_sub['input_text'][i][:250]} ...")
    print(f"     PREDICTED: {predictions[i]}")
    print(f"     REFERENCE: {references[i]}")

print(f"\n✅ Phase 1 Training complete! Outputs saved to: {run_dir}")
