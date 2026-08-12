"""
scripts/evaluate.py  —  Evaluate a saved checkpoint on any test split
Usage:
    python scripts/evaluate.py --dataset cnn_dailymail --checkpoint outputs/cnn_dailymail/run/best_model
    python scripts/evaluate.py --dataset samsum --checkpoint outputs/samsum/run/best_model --samples 200
"""
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils          import set_seed, get_device, get_model_config, get_logger
from src.dataset_loader import DatasetLoader
from src.model_factory  import ModelFactory
from src.inference      import SummarizationInference
from src.evaluator      import Evaluator

logger = get_logger("evaluate")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a summarization checkpoint")
    p.add_argument("--dataset",    required=True)
    p.add_argument("--checkpoint", required=True, help="Path to saved model directory")
    p.add_argument("--split",      default="test", choices=["train", "validation", "test"])
    p.add_argument("--samples",    type=int, default=None, help="Limit evaluation samples")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--mode",       default="beam", choices=["beam", "sampling"])
    return p.parse_args()


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = get_device()
    cfg    = get_model_config(args.dataset)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer = ModelFactory.from_checkpoint(args.checkpoint, device=device)
    inferencer = SummarizationInference(model, tokenizer, cfg, device)

    # Load data
    loader = DatasetLoader(args.dataset)
    splits = loader.load()
    ds     = splits.get(args.split, splits["test"])
    if args.samples:
        ds = ds.select(range(min(args.samples, len(ds))))

    logger.info(f"Evaluating {len(ds)} samples from '{args.split}' split ...")
    predictions = inferencer.batch_summarize(
        ds["input_text"], mode=args.mode, batch_size=cfg.get("batch_size", 4)
    )
    references  = ds["target_text"]

    ev = Evaluator()
    scores = ev.compute(predictions, references)
    ev.print_scores(scores, dataset=args.dataset, model=args.checkpoint)

    # Show 5 examples
    logger.info("\n── Sample Summaries ──────────────────────────────────")
    for i in range(min(5, len(predictions))):
        print(f"\n[{i+1}] INPUT  : {ds['input_text'][i][:200]} ...")
        print(f"     PRED   : {predictions[i]}")
        print(f"     REF    : {references[i]}")


if __name__ == "__main__":
    main()
