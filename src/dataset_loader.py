"""
src/dataset_loader.py  —  Unified HuggingFace dataset loading interface
Handles all 7 datasets with column normalization and subset sampling.
"""
from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

from datasets import load_dataset, Dataset, DatasetDict

from .utils import get_logger, get_dataset_config, clean_text

logger = get_logger("dataset_loader")

# ── Column aliases ─────────────────────────────────────────────────────────
# Maps known HF column names → canonical names used throughout this project
INPUT_ALIASES  = ["article", "document", "documents", "dialogue", "text", "body"]
TARGET_ALIASES = ["highlights", "summary", "abstract", "tldr", "headline"]

CANONICAL_INPUT  = "input_text"
CANONICAL_TARGET = "target_text"


# ──────────────────────────────────────────────────────────────────────────────
class DatasetLoader:
    """
    Load, normalize and sample any of the 7 supported datasets.

    Usage:
        loader = DatasetLoader("cnn_dailymail")
        splits = loader.load()          # returns {"train": Dataset, "validation": Dataset, "test": Dataset}
        train_ds = splits["train"]
        print(train_ds[0])              # {"input_text": ..., "target_text": ...}
    """

    SUPPORTED = [
        "cnn_dailymail", "samsum", "xsum",
        "gigaword", "arxiv", "reddit_tifu", "aclsum",
    ]

    def __init__(
        self,
        dataset_name: str,
        config_path: str = "config/dataset_configs.yaml",
        seed: int = 42,
    ):
        if dataset_name not in self.SUPPORTED:
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose from: {self.SUPPORTED}")
        self.name        = dataset_name
        self.cfg         = get_dataset_config(dataset_name, config_path)
        self.seed        = seed
        self._raw: Optional[DatasetDict] = None

    # ── Public API ────────────────────────────────────────────────────────
    def load(self) -> Dict[str, Dataset]:
        """Download (cached) and return normalized train/val/test splits."""
        logger.info(f"Loading dataset: {self.name} ...")
        self._raw = self._download()
        splits    = self._build_splits()
        self._log_stats(splits)
        return splits

    # ── Download ──────────────────────────────────────────────────────────
    def _download(self) -> DatasetDict:
        hf_name   = self.cfg["hf_name"]
        hf_config = self.cfg.get("hf_config")

        try:
            if hf_config:
                return load_dataset(hf_name, hf_config, trust_remote_code=True)
            return load_dataset(hf_name, trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to download '{hf_name}': {e}")
            raise

    # ── Split construction ────────────────────────────────────────────────
    def _build_splits(self) -> Dict[str, Dataset]:
        cfg   = self.cfg
        raw   = self._raw
        out   = {}

        for split_key, n_key in [
            ("train_split", "train_samples"),
            ("val_split",   "val_samples"),
            ("test_split",  "test_samples"),
        ]:
            split_name = cfg.get(split_key)
            n          = cfg.get(n_key)

            if split_name is None:
                # Datasets like reddit_tifu have no predefined val/test → manual split
                if split_key == "train_split":
                    logger.warning(f"No train split defined for {self.name}. Skipping.")
                continue

            if split_name not in raw:
                logger.warning(f"Split '{split_name}' not in dataset. Skipping.")
                continue

            ds = raw[split_name]

            # Sub-sample if requested
            if n and n < len(ds):
                indices = random.Random(self.seed).sample(range(len(ds)), n)
                ds = ds.select(indices)

            ds = self._normalize(ds)
            label = split_key.replace("_split", "")   # "train" / "val" / "test"
            out[label] = ds

        # Handle reddit_tifu: no predefined val/test → split train 80/10/10
        if self.name == "reddit_tifu" and "validation" not in out:
            out = self._manual_split(out["train"])

        return out

    def _manual_split(self, ds: Dataset, train_frac=0.8, val_frac=0.1) -> Dict[str, Dataset]:
        """Split a single dataset into train/val/test."""
        n       = len(ds)
        indices = list(range(n))
        random.Random(self.seed).shuffle(indices)

        tr_end  = int(n * train_frac)
        va_end  = int(n * (train_frac + val_frac))

        return {
            "train":      ds.select(indices[:tr_end]),
            "validation": ds.select(indices[tr_end:va_end]),
            "test":       ds.select(indices[va_end:]),
        }

    # ── Column normalisation ──────────────────────────────────────────────
    def _normalize(self, ds: Dataset) -> Dataset:
        """Rename dataset columns to canonical input_text / target_text."""
        cfg         = self.cfg
        input_col   = cfg["input_col"]
        target_col  = cfg["target_col"]

        cols = ds.column_names

        # Rename
        rename_map = {}
        if input_col in cols and input_col != CANONICAL_INPUT:
            rename_map[input_col] = CANONICAL_INPUT
        if target_col in cols and target_col != CANONICAL_TARGET:
            rename_map[target_col] = CANONICAL_TARGET

        if rename_map:
            ds = ds.rename_columns(rename_map)

        # Drop all other columns
        keep = {CANONICAL_INPUT, CANONICAL_TARGET}
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)

        # Clean text
        ds = ds.map(
            lambda ex: {
                CANONICAL_INPUT:  clean_text(str(ex[CANONICAL_INPUT]  or "")),
                CANONICAL_TARGET: clean_text(str(ex[CANONICAL_TARGET] or "")),
            },
            desc=f"Cleaning {self.name}",
        )

        # Filter empty rows
        ds = ds.filter(
            lambda ex: len(ex[CANONICAL_INPUT]) > 10 and len(ex[CANONICAL_TARGET]) > 3
        )

        return ds

    # ── Stats ─────────────────────────────────────────────────────────────
    def _log_stats(self, splits: Dict[str, Dataset]) -> None:
        for split_name, ds in splits.items():
            n      = len(ds)
            sample = ds[0] if n > 0 else {}
            inp_l  = len(sample.get(CANONICAL_INPUT, "").split())
            tgt_l  = len(sample.get(CANONICAL_TARGET, "").split())
            logger.info(
                f"  [{self.name}] {split_name:12s} → {n:>7,} samples  "
                f"(sample input_words≈{inp_l}, target_words≈{tgt_l})"
            )

    # ── Convenience ───────────────────────────────────────────────────────
    def get_sample(self, split: str = "train", idx: int = 0) -> Dict:
        raw = self._raw
        if raw is None:
            raise RuntimeError("Call .load() first.")
        return {
            CANONICAL_INPUT:  str(raw[split][self.cfg["input_col"]][idx]),
            CANONICAL_TARGET: str(raw[split][self.cfg["target_col"]][idx]),
        }

    @property
    def phase(self) -> int:
        return self.cfg["phase"]

    @property
    def domain(self) -> str:
        return self.cfg["domain"]


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DatasetLoader("samsum")
    splits = loader.load()
    print("Train sample:\n", splits["train"][0])
