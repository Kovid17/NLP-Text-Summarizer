"""
src/model_factory.py  —  Load any BART / T5 / Pegasus / LED model + tokenizer
from HuggingFace Hub with a single call.
"""
from __future__ import annotations

from typing import Dict, Tuple, Any

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
    T5ForConditionalGeneration,
    PegasusForConditionalGeneration,
    LEDForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .utils import get_logger, get_model_config, get_device

logger = get_logger("model_factory")

# ── Explicit model-class routing (fallback: AutoModel) ───────────────────────
_MODEL_CLASS_MAP = {
    "bart":   BartForConditionalGeneration,
    "t5":     T5ForConditionalGeneration,
    "pegasus": PegasusForConditionalGeneration,
    "led":    LEDForConditionalGeneration,
}


def _infer_model_class(model_name: str):
    name_lower = model_name.lower()
    for key, cls in _MODEL_CLASS_MAP.items():
        if key in name_lower:
            return cls
    return AutoModelForSeq2SeqLM


# ──────────────────────────────────────────────────────────────────────────────
class ModelFactory:
    """
    Central factory for loading models and tokenizers.

    Usage (by dataset name — reads config automatically):
        model, tokenizer = ModelFactory.from_dataset("cnn_dailymail")

    Usage (by explicit HF model name):
        model, tokenizer = ModelFactory.from_name("facebook/bart-large-cnn")
    """

    # ── By dataset name ───────────────────────────────────────────────────
    @classmethod
    def from_dataset(
        cls,
        dataset_name: str,
        config_path: str = "config/model_configs.yaml",
        device=None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        cfg = get_model_config(dataset_name, config_path)
        model_name = cfg["model_name"]
        tokenizer_name = cfg.get("tokenizer_name", model_name)
        return cls.from_name(model_name, tokenizer_name=tokenizer_name, device=device)

    # ── By explicit name ──────────────────────────────────────────────────
    @classmethod
    def from_name(
        cls,
        model_name: str,
        tokenizer_name: str | None = None,
        device=None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if tokenizer_name is None:
            tokenizer_name = model_name

        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        logger.info(f"Loading model: {model_name}")
        model_cls = _infer_model_class(model_name)
        model = model_cls.from_pretrained(model_name)

        if device is None:
            device = get_device()

        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"  Parameters: {n_params:.1f} M  |  Device: {device}")

        return model, tokenizer

    # ── From local checkpoint ─────────────────────────────────────────────
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        device=None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        logger.info(f"Loading from checkpoint: {checkpoint_dir}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model     = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        if device is None:
            device = get_device()
        model = model.to(device)
        return model, tokenizer

    # ── Model info ────────────────────────────────────────────────────────
    @staticmethod
    def model_info(model: PreTrainedModel) -> Dict[str, Any]:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total_params_M":     round(total     / 1e6, 2),
            "trainable_params_M": round(trainable / 1e6, 2),
            "architecture":       model.__class__.__name__,
        }


# ── Dataset → model name mapping (for quick reference / UI) ──────────────────
DATASET_MODEL_MAP = {
    "cnn_dailymail": "facebook/bart-large-cnn",
    "samsum":        "philschmid/bart-large-cnn-samsum",
    "xsum":          "facebook/bart-large-xsum",
    "gigaword":      "google/pegasus-gigaword",
    "arxiv":         "google/pegasus-arxiv",
    "reddit_tifu":   "facebook/bart-large",
    "aclsum":        "allenai/led-base-16384",
}


def get_pretrained_model_name(dataset_name: str) -> str:
    """Return the recommended pre-trained HF model name for a given dataset."""
    if dataset_name not in DATASET_MODEL_MAP:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Options: {list(DATASET_MODEL_MAP)}")
    return DATASET_MODEL_MAP[dataset_name]
