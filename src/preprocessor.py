"""
src/preprocessor.py  —  Tokenization, padding, truncation & DataLoader creation
Works with any HuggingFace Seq2Seq tokenizer (BART, T5, Pegasus, LED).
"""
from __future__ import annotations

from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .utils import get_logger

logger = get_logger("preprocessor")

CANONICAL_INPUT  = "input_text"
CANONICAL_TARGET = "target_text"


# ──────────────────────────────────────────────────────────────────────────────
class Preprocessor:
    """
    Tokenizes input/target pairs and wraps them in PyTorch DataLoaders.

    Usage:
        pre = Preprocessor(tokenizer, model_cfg)
        train_loader = pre.get_dataloader(train_ds, shuffle=True)
        val_loader   = pre.get_dataloader(val_ds,   shuffle=False)
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, model_cfg: Dict[str, Any]):
        self.tokenizer       = tokenizer
        self.max_input_len   = model_cfg.get("max_input_length",  512)
        self.max_target_len  = model_cfg.get("max_target_length", 128)
        self.batch_size      = model_cfg.get("batch_size",         4)
        self.is_t5           = "t5" in tokenizer.name_or_path.lower()
        self.is_led          = "led" in tokenizer.name_or_path.lower()

    # ── Tokenise a HuggingFace Dataset ───────────────────────────────────
    def tokenize_dataset(self, dataset: Dataset, desc: str = "Tokenizing") -> Dataset:
        """
        Map raw text Dataset → tokenized Dataset with input_ids, attention_mask,
        labels (and global_attention_mask for LED models).
        """
        tokenized = dataset.map(
            self._tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names,
            desc=desc,
        )
        tokenized.set_format("torch")
        return tokenized

    def _tokenize_batch(self, batch: Dict) -> Dict:
        inputs  = batch[CANONICAL_INPUT]
        targets = batch[CANONICAL_TARGET]

        # T5 models expect a task prefix
        if self.is_t5:
            inputs = ["summarize: " + t for t in inputs]

        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,   # return lists; DataLoader collates later
        )

        # Tokenize targets (labels)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.max_target_len,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )

        # Replace padding token id with -100 so loss ignores them
        label_ids = [
            [(lbl if lbl != self.tokenizer.pad_token_id else -100) for lbl in seq]
            for seq in labels["input_ids"]
        ]

        model_inputs["labels"] = label_ids

        # LED: set global attention on [CLS] token (first token)
        if self.is_led:
            gam = [[0] * len(mask) for mask in model_inputs["attention_mask"]]
            for g in gam:
                if g:
                    g[0] = 1
            model_inputs["global_attention_mask"] = gam

        return model_inputs

    # ── DataLoader ────────────────────────────────────────────────────────
    def get_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        """Return a PyTorch DataLoader for the given (already tokenized) Dataset."""
        tokenized = self.tokenize_dataset(dataset)
        return DataLoader(
            tokenized,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    @staticmethod
    def _collate_fn(batch):
        """Stack list of dicts → dict of tensors."""
        keys = batch[0].keys()
        return {k: torch.stack([torch.tensor(b[k]) for b in batch]) for k in keys}

    # ── Quick stats ───────────────────────────────────────────────────────
    def describe(self, dataset: Dataset, n_samples: int = 1000) -> Dict[str, float]:
        """Return average input/target token lengths for a dataset sample."""
        import numpy as np
        sample = dataset.select(range(min(n_samples, len(dataset))))
        inp_lens, tgt_lens = [], []
        for row in sample:
            inp_lens.append(len(self.tokenizer(row[CANONICAL_INPUT])["input_ids"]))
            tgt_lens.append(len(self.tokenizer(row[CANONICAL_TARGET])["input_ids"]))
        return {
            "avg_input_tokens":  float(np.mean(inp_lens)),
            "max_input_tokens":  int(np.max(inp_lens)),
            "avg_target_tokens": float(np.mean(tgt_lens)),
            "max_target_tokens": int(np.max(tgt_lens)),
        }
