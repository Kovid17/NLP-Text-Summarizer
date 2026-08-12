"""
data/download_datasets.py  —  Download & cache all 7 datasets from HuggingFace
Run: python data/download_datasets.py --phase all
"""
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset_loader import DatasetLoader
from src.utils import get_logger

logger = get_logger("download_datasets")

PHASE_MAP = {
    1: ["cnn_dailymail", "samsum"],
    2: ["xsum", "gigaword"],
    3: ["arxiv", "reddit_tifu", "aclsum"],
}

def download(phase: str = "all"):
    datasets = (
        [ds for p in PHASE_MAP.values() for ds in p]
        if phase == "all"
        else PHASE_MAP.get(int(phase), [])
    )
    for name in datasets:
        try:
            logger.info(f"Downloading: {name} ...")
            splits = DatasetLoader(name).load()
            total  = sum(len(v) for v in splits.values())
            logger.info(f"  ✅ {name} — {total:,} samples cached")
        except Exception as e:
            logger.error(f"  ❌ {name}: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="all")
    download(p.parse_args().phase)
