"""
scripts/zero_shot_baseline.py  —  Run pre-trained models (no fine-tuning) on all datasets
Establishes baseline ROUGE scores before any training.
Usage:
    python scripts/zero_shot_baseline.py --phase 1
    python scripts/zero_shot_baseline.py --phase all
"""
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils          import set_seed, get_device, get_model_config, get_logger
from src.dataset_loader import DatasetLoader
from src.model_factory  import ModelFactory, DATASET_MODEL_MAP
from src.inference      import SummarizationInference
from src.evaluator      import Evaluator

logger = get_logger("zero_shot")

PHASE_MAP = {
    1: ["cnn_dailymail", "samsum"],
    2: ["xsum", "gigaword"],
    3: ["arxiv", "reddit_tifu", "aclsum"],
}

N_EVAL = 100   # samples per dataset for quick baseline


def run_baseline(dataset_name: str, device):
    logger.info(f"\n{'─'*55}")
    logger.info(f"  BASELINE: {dataset_name}")
    logger.info(f"{'─'*55}")

    cfg = get_model_config(dataset_name)
    try:
        model, tokenizer = ModelFactory.from_dataset(dataset_name, device=device)
    except Exception as e:
        logger.error(f"  Could not load model: {e}")
        return None

    splits = DatasetLoader(dataset_name).load()
    ds     = splits.get("test", splits.get("validation", splits["train"]))
    ds     = ds.select(range(min(N_EVAL, len(ds))))

    inferencer  = SummarizationInference(model, tokenizer, cfg, device)
    predictions = inferencer.batch_summarize(ds["input_text"], mode="beam")
    references  = ds["target_text"]

    ev     = Evaluator()
    scores = ev.compute(predictions, references)
    ev.print_scores(scores, dataset=dataset_name, model=cfg["model_name"])
    ev.save_csv(scores, dataset_name=dataset_name,
                model_name=f"{cfg['model_name']} (zero-shot)",
                output_dir="reports/results")
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="1", help="1 | 2 | 3 | all")
    p.add_argument("--seed",  type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()

    datasets = (
        [ds for ph in PHASE_MAP.values() for ds in ph]
        if args.phase == "all"
        else PHASE_MAP.get(int(args.phase), [])
    )

    all_scores = {}
    for name in datasets:
        scores = run_baseline(name, device)
        if scores:
            all_scores[name] = scores

    logger.info("\n\n══ ZERO-SHOT BASELINE SUMMARY ══")
    for name, sc in all_scores.items():
        logger.info(f"  {name:20s} R1={sc['rouge1']:5.2f}  R2={sc['rouge2']:5.2f}  RL={sc['rougeL']:5.2f}")


if __name__ == "__main__":
    main()
