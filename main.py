#!/usr/bin/env python3
"""
Main pipeline for STR (Short Tandem Repeat) classification from BAM files.

This pipeline:
1. Extracts reads from BAM files (local or remote URLs)
2. Processes sequences to extract features (k-mers, CIGAR, TRMotif, STRling)
3. Trains XGBoost/LightGBM/RF classifier to identify STRs
4. Calibrates probabilities and optimises threshold
5. Annotates predictions with REDatlas disease associations
6. Saves results, predictions, and summaries
"""

import sys
import argparse
import logging
from pathlib import Path

from src.input.bam_process import extract_sequences_from_bam
from src.utils.model_utils import train_str_classifier, predict_str_sequences
from src.utils.data_utils import (
    load_labeled_data,
    create_balanced_dataset,
    load_sequences_for_prediction,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_yaml_config(config_path: str) -> dict:
    """Load YAML config file."""
    try:
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Could not load config {config_path}: {e}")
        return {}


def run_pipeline(args):
    """Run the complete STR classification pipeline."""

    logger.info("=" * 80)
    logger.info("STR CLASSIFICATION PIPELINE")
    logger.info("=" * 80)

    # Load config if available
    config_path = args.config
    cfg = _load_yaml_config(config_path) if Path(config_path).exists() else {}
    model_cfg = cfg.get("model", {})

    # Step 1: Load or extract training data
    if args.str_data and args.normal_data:
        logger.info("[Step 1] Loading pre-labeled data...")
        str_sequences = load_labeled_data(args.str_data)
        normal_sequences = load_labeled_data(args.normal_data)
        training_data = create_balanced_dataset(str_sequences, normal_sequences)

    elif args.bam_file:
        logger.info("[Step 1] Extracting sequences from BAM file...")
        output_path = f"{args.output_dir}/extracted_sequences.json"
        training_data = extract_sequences_from_bam(
            bam_file=args.bam_file,
            chromosome=args.chromosome,
            start=args.start,
            end=args.end,
            max_reads=args.max_reads,
            output_file=output_path,
        )

        if not training_data:
            logger.error("No training data available")
            return 1

        logger.info("Sequences extracted but not labeled. Please label them manually.")
        logger.info("Set 'is_str': true or false for each sequence.")
        return 0

    else:
        logger.error("Must provide either:")
        logger.error("  - Pre-labeled data (--str-data and --normal-data)")
        logger.error("  - BAM file for extraction (--bam-file)")
        return 1

    # Step 2: Train classifier (or load pre-trained model)
    from src.model.str_classifier import STRClassifier

    if args.model_path and Path(args.model_path).exists():
        logger.info(f"[Step 2] Loading pre-trained model from {args.model_path}")
        classifier = STRClassifier.load(args.model_path)
    elif args.train:
        logger.info("[Step 2] Training STR classifier...")
        train_result = train_str_classifier(
            training_data=training_data,
            test_size=model_cfg.get("test_size", args.test_size),
            cv_folds=model_cfg.get("cv_folds", args.cv_folds),
            threshold=model_cfg.get("threshold", args.threshold),
            output_dir=args.output_dir,
            config_path=config_path,
        )
        classifier = train_result["classifier"]
    else:
        logger.info("[Step 2] Skipping training (use --train flag or --model-path)")
        return 0

    # Step 3: Make predictions on new data if provided
    if args.predict_file:
        logger.info("[Step 3] Making predictions on new data...")
        predict_sequences = load_sequences_for_prediction(args.predict_file)
        predict_str_sequences(
            classifier=classifier,
            sequences=predict_sequences,
            output_path=f"{args.output_dir}/predictions.json",
            config_path=config_path,
        )
    elif args.predict_bam:
        logger.info("[Step 3] Making predictions on BAM file...")
        predict_sequences = extract_sequences_from_bam(
            bam_file=args.predict_bam,
            chromosome=args.predict_chr,
            start=args.predict_start,
            end=args.predict_end,
            max_reads=args.predict_max_reads,
        )
        if predict_sequences:
            predict_str_sequences(
                classifier=classifier,
                sequences=predict_sequences,
                output_path=f"{args.output_dir}/predictions.json",
                config_path=config_path,
            )
        else:
            logger.warning("No sequences extracted for prediction")

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="STR Classification Pipeline - Train and predict Short Tandem Repeats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract sequences from BAM file for labeling
  python main.py --bam-file file.bam --chromosome chr20 --start 10000000 --end 10100000

  # Train classifier with pre-labeled data (uses config.yaml for model settings)
  python main.py --str-data output/str_variants.json --normal-data output/normal_sequences.json --train

  # Train and predict on new BAM file
  python main.py --str-data output/str_variants.json --normal-data output/normal_sequences.json --train \\
                 --predict-bam file.bam --predict-chr chr20 --predict-start 10000000 --predict-end 10100000

  # Predict using a pre-trained model
  python main.py --model-path output/str_model.joblib --predict-file sequences.json
        """,
    )

    # Configuration
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )

    # Input data arguments
    input_group = parser.add_argument_group("Input Data")
    input_group.add_argument("--str-data", help="JSON file with STR sequences")
    input_group.add_argument("--normal-data", help="JSON file with normal (non-STR) sequences")
    input_group.add_argument("--bam-file", help="BAM file path or URL for extraction")
    input_group.add_argument("--chromosome", default="chr20", help="Chromosome (default: chr20)")
    input_group.add_argument("--start", type=int, default=10000000, help="Start position")
    input_group.add_argument("--end", type=int, default=10100000, help="End position")
    input_group.add_argument("--max-reads", type=int, default=1000, help="Max reads to extract")

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--train", action="store_true", help="Train the classifier")
    train_group.add_argument("--model-path", help="Path to pre-trained model (.joblib)")
    train_group.add_argument("--test-size", type=float, default=0.2, help="Test set size")
    train_group.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    train_group.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")

    # Prediction arguments
    pred_group = parser.add_argument_group("Prediction")
    pred_group.add_argument("--predict-file", help="JSON file with sequences to predict")
    pred_group.add_argument("--predict-bam", help="BAM file for prediction")
    pred_group.add_argument("--predict-chr", default="chr20", help="Chromosome for prediction")
    pred_group.add_argument("--predict-start", type=int, default=10000000)
    pred_group.add_argument("--predict-end", type=int, default=10100000)
    pred_group.add_argument("--predict-max-reads", type=int, default=1000)

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--output-dir", default="output", help="Output directory")

    args = parser.parse_args()
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
