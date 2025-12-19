#!/usr/bin/env python3
"""
Main pipeline for STR (Short Tandem Repeat) classification from BAM files.

This pipeline:
1. Extracts reads from BAM files (local or remote URLs)
2. Processes sequences to extract features
3. Trains a Random Forest classifier to identify STRs
4. Evaluates model performance
5. Saves results and predictions
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pysam

from src.input.bam_process import BamRegionMetadata, extract_bam_region, extract_sequences_from_bam, parse_read_to_metadata
from src.model.str_classifier import STR_Classifier

from src.utils.model_utils import train_str_classifier, predict_str_sequences
from src.utils.data_utils import load_labeled_data, create_balanced_dataset, load_sequences_for_prediction


def run_pipeline(args):
    """Run the complete STR classification pipeline."""
    
    print("\n" + "="*80)
    print("STR CLASSIFICATION PIPELINE")
    print("="*80)
    
    # Step 1: Load or extract training data
    if args.str_data and args.normal_data:
        print("\n[Step 1] Loading pre-labeled data...")
        str_sequences = load_labeled_data(args.str_data)
        normal_sequences = load_labeled_data(args.normal_data)
        training_data = create_balanced_dataset(str_sequences, normal_sequences)
    
    elif args.bam_file:
        print("\n[Step 1] Extracting sequences from BAM file...")
        training_data = extract_sequences_from_bam(
            bam_file=args.bam_file,
            chromosome=args.chromosome,
            start=args.start,
            end=args.end,
            max_reads=args.max_reads
        )
        
        if not training_data:
            print("ERROR: No training data available")
            return 1
        
        # Save extracted sequences
        with open("output/extracted_sequences.json", 'w') as f:
            json.dump(training_data, f, indent=2)
        print("NOTE: Sequences extracted but not labeled. Please label them manually.")
        print("      Set 'is_str': true or false for each sequence.")
        return 0
    
    else:
        print("ERROR: Must provide either:")
        print("  - Pre-labeled data (--str-data and --normal-data)")
        print("  - BAM file for extraction (--bam-file)")
        return 1
    
    # Step 2: Train classifier
    if args.train:
        print("\n[Step 2] Training STR classifier...")
        train_result = train_str_classifier(
            training_data=training_data,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            threshold=args.threshold,
            output_dir=args.output_dir
        )
        classifier = train_result['classifier']
    else:
        print("\n[Step 2] Skipping training (use --train flag)")
        return 0
    
    # Step 3: Make predictions on new data if provided
    if args.predict_file:
        print("\n[Step 3] Making predictions on new data with motif detection...")
        predict_sequences = load_sequences_for_prediction(args.predict_file)
        predictions = predict_str_sequences(
            classifier=classifier,
            sequences=predict_sequences,
            output_path=f"{args.output_dir}/predictions.json"
        )
    elif args.predict_bam:
        print("\n[Step 3] Making predictions on BAM file with motif detection...")
        predict_sequences = extract_sequences_from_bam(
            bam_file=args.predict_bam,
            chromosome=args.predict_chr,
            start=args.predict_start,
            end=args.predict_end,
            max_reads=args.predict_max_reads
        )
        if predict_sequences:
            predictions = predict_str_sequences(
                classifier=classifier,
                sequences=predict_sequences,
                output_path=f"{args.output_dir}/predictions.json"
            )
        else:
            print("WARNING: No sequences extracted for prediction")
    
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="STR Classification Pipeline - Train and predict Short Tandem Repeats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract sequences from BAM file for labeling (larger region)
  python main.py --bam-file file.bam --chromosome chr20 --start 10000000 --end 10100000
  
  # Train classifier with pre-labeled data
  python main.py --str-data output/str_variants.json --normal-data output/normal_sequences.json --train
  
  # Train and predict on new BAM file
  python main.py --str-data output/str_variants.json --normal-data output/normal_sequences.json --train \\
                 --predict-bam file.bam --predict-chr chr20 --predict-start 10000000 --predict-end 10100000
        """
    )
    
    # Input data arguments
    input_group = parser.add_argument_group('Input Data')
    input_group.add_argument('--str-data', help='JSON file with STR sequences')
    input_group.add_argument('--normal-data', help='JSON file with normal (non-STR) sequences')
    input_group.add_argument('--bam-file', help='BAM file path or URL for extraction')
    input_group.add_argument('--chromosome', default='chr20', help='Chromosome (default: chr20)')
    input_group.add_argument('--start', type=int, default=10000000, help='Start position (default: 10000000)')
    input_group.add_argument('--end', type=int, default=10100000, help='End position (default: 10100000)')
    input_group.add_argument('--max-reads', type=int, default=1000, help='Max reads to extract')
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--train', action='store_true', help='Train the classifier')
    train_group.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    train_group.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds (default: 5)')
    train_group.add_argument('--threshold', type=float, default=0.5, help='Classification threshold (default: 0.5)')
    
    # Prediction arguments
    pred_group = parser.add_argument_group('Prediction')
    pred_group.add_argument('--predict-file', help='JSON file with sequences to predict')
    pred_group.add_argument('--predict-bam', help='BAM file for prediction')
    pred_group.add_argument('--predict-chr', default='chr20', help='Chromosome for prediction')
    pred_group.add_argument('--predict-start', type=int, default=10000000, help='Start position for prediction')
    pred_group.add_argument('--predict-end', type=int, default=10100000, help='End position for prediction')
    pred_group.add_argument('--predict-max-reads', type=int, default=1000, help='Max reads for prediction')
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', default='output', help='Output directory (default: output)')
    
    args = parser.parse_args()
    
    # Run pipeline
    return run_pipeline(args)



if __name__ == "__main__":
    sys.exit(main())