"""
src/evaluator.py  —  ROUGE-1/2/L, BLEU, METEOR evaluation + report generation
"""
from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional

import evaluate
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import get_logger

logger = get_logger("evaluator")

# Download NLTK data silently once
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)


# ──────────────────────────────────────────────────────────────────────────────
class Evaluator:
    """
    Compute and report summarization metrics.

    Usage:
        ev = Evaluator()
        scores = ev.compute(predictions=["..."], references=["..."])
        print(scores)
        # {"rouge1": 0.44, "rouge2": 0.21, "rougeL": 0.41, "bleu": 18.5, "meteor": 0.32}
    """

    def __init__(self):
        self._rouge  = evaluate.load("rouge")
        self._bleu   = evaluate.load("bleu")
        self._meteor = evaluate.load("meteor")

    # ── Core compute ──────────────────────────────────────────────────────
    def compute(
        self,
        predictions: List[str],
        references:  List[str],
        use_stemmer: bool = True,
    ) -> Dict[str, float]:
        """Return dict of metric_name → score (0–100 for ROUGE/METEOR, raw for BLEU)."""
        assert len(predictions) == len(references), \
            "predictions and references must have the same length"

        results: Dict[str, float] = {}

        # ROUGE
        rouge_out = self._rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=use_stemmer,
        )
        results["rouge1"] = round(rouge_out["rouge1"] * 100, 2)
        results["rouge2"] = round(rouge_out["rouge2"] * 100, 2)
        results["rougeL"] = round(rouge_out["rougeL"] * 100, 2)

        # BLEU  (expects list of reference strings)
        try:
            bleu_out = self._bleu.compute(
                predictions=predictions,
                references=[[r] for r in references],
            )
            results["bleu"] = round(bleu_out["bleu"] * 100, 2)
        except Exception:
            results["bleu"] = 0.0

        # METEOR
        try:
            meteor_out = self._meteor.compute(
                predictions=predictions,
                references=references,
            )
            results["meteor"] = round(meteor_out["meteor"] * 100, 2)
        except Exception:
            results["meteor"] = 0.0

        return results

    # ── Compute per-sample ROUGE-L ────────────────────────────────────────
    def per_sample_rougeL(
        self,
        predictions: List[str],
        references:  List[str],
    ) -> List[float]:
        scores = []
        for p, r in zip(predictions, references):
            s = self._rouge.compute(predictions=[p], references=[r])
            scores.append(round(s["rougeL"] * 100, 2))
        return scores

    # ── Save results ──────────────────────────────────────────────────────
    def save_csv(
        self,
        results: Dict[str, float],
        dataset_name: str,
        model_name:   str,
        output_dir:   str = "reports/results",
    ) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(output_dir, "all_results.csv")
        row  = {"dataset": dataset_name, "model": model_name, **results}

        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        logger.info(f"Results saved → {path}")
        return path

    # ── Visualisation ─────────────────────────────────────────────────────
    def plot_scores(
        self,
        results_csv:  str = "reports/results/all_results.csv",
        output_dir:   str = "reports/figures",
        metric:       str = "rouge1",
    ) -> str:
        """Bar chart: metric score per dataset × model."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(results_csv)

        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, x="dataset", y=metric, hue="model", palette="viridis")
        ax.set_title(f"{metric.upper()} Scores by Dataset & Model", fontsize=15, fontweight="bold")
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel(f"{metric.upper()} (×100)", fontsize=12)
        plt.xticks(rotation=30, ha="right")
        plt.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"{metric}_comparison.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Plot saved → {out_path}")
        return out_path

    def plot_radar(
        self,
        results: Dict[str, float],
        title:   str = "Model Performance",
        output_dir: str = "reports/figures",
    ) -> str:
        """Radar/spider chart for a single model's metric scores."""
        import numpy as np
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        metrics = ["rouge1", "rouge2", "rougeL", "bleu", "meteor"]
        values  = [results.get(m, 0) for m in metrics]
        values += values[:1]   # close the polygon

        angles  = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, alpha=0.25, color="#4C72B0")
        ax.plot(angles, values, color="#4C72B0", linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics], fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()

        safe_title = title.replace(" ", "_").replace("/", "-")
        out_path   = os.path.join(output_dir, f"radar_{safe_title}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path

    # ── Pretty print ──────────────────────────────────────────────────────
    @staticmethod
    def print_scores(scores: Dict[str, float], dataset: str = "", model: str = "") -> None:
        header = f"{'─'*50}"
        print(header)
        if dataset or model:
            print(f"  Dataset: {dataset}   |   Model: {model}")
        for k, v in scores.items():
            print(f"  {k.upper():10s}: {v:.2f}")
        print(header)
