"""
src/trainer.py  —  Training loop with ROUGE validation + checkpointing
Supports: BART, T5, Pegasus, LED (Seq2Seq Transformer)
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from .evaluator import Evaluator
from .utils import get_logger, save_metrics

logger = get_logger("trainer")


# ──────────────────────────────────────────────────────────────────────────────
class Trainer:
    """
    Full training + validation loop for Seq2Seq summarization.

    Usage:
        trainer = Trainer(model, tokenizer, model_cfg, device, output_dir)
        trainer.train(train_loader, val_loader, val_dataset)
    """

    def __init__(
        self,
        model:       PreTrainedModel,
        tokenizer:   PreTrainedTokenizerBase,
        model_cfg:   Dict[str, Any],
        device:      torch.device,
        output_dir:  str = "outputs/run",
    ):
        self.model       = model
        self.tokenizer   = tokenizer
        self.cfg         = model_cfg
        self.device      = device
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator   = Evaluator()
        self.best_rouge  = -1.0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_rouge1": [], "val_rouge2": [], "val_rougeL": []
        }

        # Scaler for FP16 / mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=(
            model_cfg.get("fp16", False) and torch.cuda.is_available()
        ))
        self.use_fp16 = model_cfg.get("fp16", False) and torch.cuda.is_available()

    # ── Main entry point ──────────────────────────────────────────────────
    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        val_dataset,                    # raw HF Dataset for decoding during eval
        num_epochs:   Optional[int] = None,
    ) -> Dict[str, list]:
        """Run training for num_epochs epochs and return history dict."""
        epochs     = num_epochs or self.cfg.get("num_epochs", 3)
        grad_accum = self.cfg.get("gradient_accumulation_steps", 1)

        total_steps = (len(train_loader) // grad_accum) * epochs
        warmup      = self.cfg.get("warmup_steps", 500)

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.get("learning_rate", 3e-5),
            weight_decay=self.cfg.get("weight_decay", 0.01),
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )

        logger.info(f"Training for {epochs} epochs | {total_steps} total steps | "
                    f"FP16={self.use_fp16}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, grad_accum)
            val_scores = self._validate(val_loader, val_dataset)

            self.history["train_loss"].append(train_loss)
            self.history["val_rouge1"].append(val_scores["rouge1"])
            self.history["val_rouge2"].append(val_scores["rouge2"])
            self.history["val_rougeL"].append(val_scores["rougeL"])

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch}/{epochs} | loss={train_loss:.4f} | "
                f"R1={val_scores['rouge1']:.2f} R2={val_scores['rouge2']:.2f} "
                f"RL={val_scores['rougeL']:.2f} | {elapsed/60:.1f} min"
            )

            # Save best checkpoint
            if val_scores["rouge1"] > self.best_rouge:
                self.best_rouge = val_scores["rouge1"]
                self._save_checkpoint(epoch, val_scores)
                logger.info(f"  ✅ New best ROUGE-1: {self.best_rouge:.2f} — checkpoint saved")

            # Early stopping
            if self._should_stop():
                logger.info("Early stopping triggered.")
                break

        save_metrics(self.history, str(self.output_dir / "training_history.yaml"))
        return self.history

    # ── Train one epoch ───────────────────────────────────────────────────
    def _train_epoch(
        self,
        loader:     DataLoader,
        optimizer:  AdamW,
        scheduler,
        grad_accum: int,
    ) -> float:
        self.model.train()
        total_loss  = 0.0
        step_count  = 0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc="  Training", leave=False)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=self.use_fp16):
                outputs = self.model(**batch)
                loss    = outputs.loss / grad_accum

            self.scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

            total_loss += loss.item() * grad_accum
            pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")

        return total_loss / max(len(loader), 1)

    # ── Validation ────────────────────────────────────────────────────────
    def _validate(self, loader: DataLoader, val_dataset) -> Dict[str, float]:
        self.model.eval()
        predictions, references = [], []

        num_beams   = self.cfg.get("num_beams",     4)
        max_tgt_len = self.cfg.get("max_target_length", 128)

        with torch.no_grad():
            for batch in tqdm(loader, desc="  Validating", leave=False):
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tgt_len,
                    num_beams=num_beams,
                    early_stopping=True,
                )
                decoded_preds = self.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                predictions.extend(decoded_preds)

        # Get references from raw dataset
        from .dataset_loader import CANONICAL_TARGET
        references = val_dataset[CANONICAL_TARGET][: len(predictions)]

        scores = self.evaluator.compute(predictions, references)
        return scores

    # ── Checkpoint ────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, scores: Dict[str, float]) -> None:
        ckpt_dir = self.output_dir / "best_model"
        self.model.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))
        save_metrics({"epoch": epoch, **scores}, str(self.output_dir / "best_scores.yaml"))

    # ── Early stopping ────────────────────────────────────────────────────
    def _should_stop(self, patience: int = 3) -> bool:
        r1_history = self.history["val_rouge1"]
        if len(r1_history) < patience + 1:
            return False
        recent = r1_history[-(patience + 1):]
        return all(recent[i] >= recent[i + 1] for i in range(patience))
