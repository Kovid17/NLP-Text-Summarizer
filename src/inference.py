"""
src/inference.py  —  Beam search & sampling inference for all models
Supports single text, batch, and streaming modes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .utils import get_logger, get_device, clean_text

logger = get_logger("inference")


# ──────────────────────────────────────────────────────────────────────────────
class SummarizationInference:
    """
    High-level inference interface for any Seq2Seq summarization model.

    Usage:
        si = SummarizationInference(model, tokenizer, model_cfg)
        summary = si.summarize("Long article text ...")
        summaries = si.batch_summarize(["text1", "text2"])
    """

    def __init__(
        self,
        model:     PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        model_cfg: Dict[str, Any],
        device:    Optional[torch.device] = None,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.cfg       = model_cfg
        self.device    = device or get_device()
        self.model     = self.model.to(self.device)
        self.model.eval()

        self.is_t5   = "t5"  in tokenizer.name_or_path.lower()
        self.is_led  = "led" in tokenizer.name_or_path.lower()

    # ── Single text ───────────────────────────────────────────────────────
    def summarize(
        self,
        text:           str,
        mode:           str = "beam",           # "beam" | "sampling"
        num_beams:      int = 4,
        temperature:    float = 0.7,
        top_p:          float = 0.9,
        max_new_tokens: Optional[int] = None,
        min_length:     int = 10,
    ) -> str:
        """Generate a summary for a single input text."""
        text  = clean_text(text)
        if not text:
            return ""

        inputs = self._encode([text])

        gen_kwargs = self._build_gen_kwargs(
            mode=mode,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens or self.cfg.get("max_target_length", 128),
            min_length=min_length,
        )

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary.strip()

    # ── Batch ─────────────────────────────────────────────────────────────
    def batch_summarize(
        self,
        texts:          List[str],
        mode:           str = "beam",
        num_beams:      int = 4,
        temperature:    float = 0.7,
        top_p:          float = 0.9,
        max_new_tokens: Optional[int] = None,
        batch_size:     int = 8,
        show_progress:  bool = True,
    ) -> List[str]:
        """Generate summaries for a list of texts in batches."""
        from tqdm import tqdm

        all_summaries = []
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        it = tqdm(batches, desc="Generating summaries") if show_progress else batches
        for batch in it:
            clean_batch = [clean_text(t) for t in batch]
            inputs = self._encode(clean_batch)
            gen_kwargs = self._build_gen_kwargs(
                mode=mode,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens or self.cfg.get("max_target_length", 128),
            )
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            all_summaries.extend([s.strip() for s in decoded])

        return all_summaries

    # ── Compare modes ─────────────────────────────────────────────────────
    def compare_modes(self, text: str) -> Dict[str, str]:
        """Return both beam-search and sampling summaries for comparison."""
        return {
            "beam_search": self.summarize(text, mode="beam", num_beams=4),
            "sampling":    self.summarize(text, mode="sampling", temperature=0.7, top_p=0.9),
        }

    # ── Encode helper ─────────────────────────────────────────────────────
    def _encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        if self.is_t5:
            texts = ["summarize: " + t for t in texts]

        max_len = self.cfg.get("max_input_length", 512)
        encoded = self.tokenizer(
            texts,
            max_length=max_len,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # LED: global attention on [CLS]
        if self.is_led and "attention_mask" in encoded:
            global_attn = torch.zeros_like(encoded["attention_mask"])
            global_attn[:, 0] = 1
            encoded["global_attention_mask"] = global_attn

        return encoded

    # ── Gen kwargs builder ────────────────────────────────────────────────
    def _build_gen_kwargs(
        self,
        mode:           str,
        num_beams:      int,
        temperature:    float,
        top_p:          float,
        max_new_tokens: int,
        min_length:     int = 10,
    ) -> Dict[str, Any]:
        base = {
            "max_new_tokens": max_new_tokens,
            "min_length":     min_length,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
        }

        if mode == "beam":
            base.update({"num_beams": num_beams, "do_sample": False})
        elif mode == "sampling":
            base.update({
                "num_beams": 1,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'beam' or 'sampling'.")

        return base

    # ── Convenience: load and infer in one call ───────────────────────────
    @classmethod
    def from_dataset(
        cls,
        dataset_name: str,
        device=None,
    ) -> "SummarizationInference":
        from .model_factory import ModelFactory
        from .utils import get_model_config
        model, tokenizer = ModelFactory.from_dataset(dataset_name, device=device)
        cfg = get_model_config(dataset_name)
        return cls(model, tokenizer, cfg, device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        model_cfg: Dict[str, Any],
        device=None,
    ) -> "SummarizationInference":
        from .model_factory import ModelFactory
        model, tokenizer = ModelFactory.from_checkpoint(checkpoint_dir, device=device)
        return cls(model, tokenizer, model_cfg, device)
