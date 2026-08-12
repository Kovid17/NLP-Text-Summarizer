"""
scripts/train.py  —  Universal training entry point for any dataset/phase
Usage:
    python scripts/train.py --dataset cnn_dailymail
    python scripts/train.py --dataset samsum --epochs 5
    python scripts/train.py --dataset xsum --fp16
"""
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils          import set_seed, get_device, get_model_config, get_run_dir, get_logger
from src.dataset_loader import DatasetLoader
from src.preprocessor   import Preprocessor
from src.model_factory  import ModelFactory
from src.trainer        import Trainer
from src.evaluator      import Evaluator

logger = get_logger("train")


def parse_args():
    p = argparse.ArgumentParser(description="Train a summarization model")
    p.add_argument("--dataset",  required=True,
                   choices=["cnn_dailymail","samsum","xsum","gigaword","arxiv","reddit_tifu","aclsum"])
    p.add_argument("--epochs",   type=int,   default=None, help="Override num_epochs in config")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--fp16",     action="store_true", help="Force FP16 mixed precision")
    p.add_argument("--output",   default="outputs",   help="Base output directory")
    return p.parse_args()


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = get_device()

    # ── Load config ──────────────────────────────────────────────────────
    cfg = get_model_config(args.dataset)
    if args.fp16:
        cfg["fp16"] = True
    if args.epochs:
        cfg["num_epochs"] = args.epochs

    logger.info(f"{'='*60}")
    logger.info(f"  Dataset : {args.dataset}")
    logger.info(f"  Model   : {cfg['model_name']}")
    logger.info(f"  Epochs  : {cfg['num_epochs']}")
    logger.info(f"  Device  : {device}")
    logger.info(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────────────
    loader = DatasetLoader(args.dataset)
    splits = loader.load()
    train_ds = splits["train"]
    val_ds   = splits.get("validation", splits["train"].select(range(min(500, len(splits["train"])))))

    # ── Model & tokenizer ────────────────────────────────────────────────
    model, tokenizer = ModelFactory.from_dataset(args.dataset, device=device)

    # ── Preprocessor & DataLoaders ───────────────────────────────────────
    pre          = Preprocessor(tokenizer, cfg)
    train_loader = pre.get_dataloader(train_ds, shuffle=True)
    val_loader   = pre.get_dataloader(val_ds,   shuffle=False)

    # ── Train ────────────────────────────────────────────────────────────
    run_dir = get_run_dir(args.dataset, base=args.output)
    trainer = Trainer(model, tokenizer, cfg, device, output_dir=str(run_dir))
    history = trainer.train(train_loader, val_loader, val_ds)

    # ── Final evaluation on test set ─────────────────────────────────────
    if "test" in splits:
        logger.info("\nRunning final evaluation on TEST set ...")
        test_ds     = splits["test"]
        test_loader = pre.get_dataloader(test_ds, shuffle=False)

        from src.inference import SummarizationInference
        inferencer = SummarizationInference(model, tokenizer, cfg, device)
        test_texts = test_ds["input_text"]
        predictions = inferencer.batch_summarize(test_texts, batch_size=cfg.get("batch_size", 4))
        references  = test_ds["target_text"]

        ev = Evaluator()
        scores = ev.compute(predictions, references)
        ev.print_scores(scores, dataset=args.dataset, model=cfg["model_name"])
        ev.save_csv(scores, dataset_name=args.dataset, model_name=cfg["model_name"],
                    output_dir=str(run_dir / "results"))

    logger.info(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
