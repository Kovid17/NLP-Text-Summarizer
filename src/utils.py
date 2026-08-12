"""
src/utils.py  —  Logging, seeding, config loading, checkpointing utilities
"""
import os
import random
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str = "nlp_summarizer", log_file: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with rich console + optional file output."""
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset_config(dataset_name: str, config_path: str = "config/dataset_configs.yaml") -> Dict:
    cfg = load_yaml_config(config_path)
    if dataset_name not in cfg:
        raise KeyError(f"Dataset '{dataset_name}' not found in {config_path}. "
                       f"Available: {list(cfg.keys())}")
    return cfg[dataset_name]


def get_model_config(dataset_name: str, config_path: str = "config/model_configs.yaml") -> Dict:
    cfg = load_yaml_config(config_path)
    if dataset_name not in cfg:
        raise KeyError(f"Model config for '{dataset_name}' not found in {config_path}.")
    return cfg[dataset_name]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_run_dir(dataset_name: str, base: str = "outputs") -> Path:
    """Create a timestamped output directory for a training run."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / dataset_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save a metrics dict as YAML."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(metrics, f, default_flow_style=False)


def load_metrics(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    """Return the best available torch device."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[Device] Using GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            device = torch.device("cpu")
            print("[Device] No GPU found — using CPU (inference only recommended)")
        return device
    except ImportError:
        raise RuntimeError("PyTorch is not installed. Run: pip install torch")


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    return len(text.split())


def truncate_text(text: str, max_words: int = 100) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def clean_text(text: str) -> str:
    """Basic text cleaning — strip, remove excess whitespace."""
    import re
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
