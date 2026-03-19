
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.model.str_classifier import STRClassifier
from src.utils.genetic_utils import deduplicate_str_predictions

logger = logging.getLogger(__name__)


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config and return the nested dict, or empty dict on failure."""
    if config_path is None:
        config_path = "config.yaml"
    p = Path(config_path)
    if not p.exists():
        return {}
    try:
        import yaml
        with open(p) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        return {}


def train_str_classifier(
    training_data: List[Dict],
    test_size: float = 0.2,
    cv_folds: int = 5,
    threshold: float = 0.5,
    output_dir: str = "output",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Train STR classifier and save results.

    Now supports config-driven algorithm selection (XGBoost / LightGBM / RF),
    Optuna hyperparameter tuning, probability calibration, and threshold
    optimisation.
    """
    cfg = _load_config(config_path)
    model_cfg = cfg.get("model", {})
    feature_cfg = cfg.get("features", {})

    algorithm = model_cfg.get("algorithm", "xgboost")
    random_state = model_cfg.get("random_state", 42)

    # Build config dict for classifier
    classifier_config = {
        "xgboost": model_cfg.get("xgboost", {}),
        "random_forest": model_cfg.get("random_forest", {}),
        "tuning": model_cfg.get("tuning", {}),
        "calibration": model_cfg.get("calibration", {}),
        "features": feature_cfg,
        "trmotif": feature_cfg.get("trmotif", {}),
        "strling_kmer": feature_cfg.get("strling_kmer", {}),
    }

    print(f"\n{'='*80}")
    print(f"TRAINING STR CLASSIFIER")
    print(f"{'='*80}")

    classifier = STRClassifier(
        threshold=threshold,
        algorithm=algorithm,
        config=classifier_config,
    )

    results = classifier.train(
        sequences=training_data,
        test_size=test_size,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_path = Path(output_dir) / "training_results.json"
    with open(results_path, "w") as f:
        json_results = {
            "accuracy": float(results["accuracy"]),
            "precision": float(results["precision"]),
            "recall": float(results["recall"]),
            "f1_score": float(results["f1_score"]),
            "specificity": float(results["specificity"]),
            "roc_auc": float(results["roc_auc"]),
            "pr_auc": float(results.get("pr_auc", 0.0)),
            "threshold": float(results.get("threshold", threshold)),
            "algorithm": results.get("algorithm", algorithm),
            "cv_folds": results["cv_folds"],
            "cv_metrics": {
                metric: [float(v) for v in values]
                for metric, values in results["cv_metrics"].items()
            },
            "feature_importance": {
                str(k): float(v)
                for k, v in results.get("feature_importance", {}).items()
            },
            "per_motif_metrics": results.get("per_motif_metrics", {}),
            "tuned_params": {
                str(k): float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in results.get("tuned_params", {}).items()
            },
        }
        json.dump(json_results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    # Save the trained model
    model_path = Path(output_dir) / "str_model.joblib"
    classifier.save(str(model_path))

    return {"classifier": classifier, "results": results}


def predict_str_sequences(
    classifier: STRClassifier,
    sequences: List[Dict],
    output_path: str = "output/predictions.json",
    config_path: Optional[str] = None,
) -> List[Dict]:
    """Make predictions with motif detection, deduplication, and REDatlas annotation."""
    cfg = _load_config(config_path)
    output_cfg = cfg.get("output", {})
    redatlas_cfg = cfg.get("redatlas", {})
    pred_cfg = cfg.get("prediction", {})

    print(f"\n{'='*80}")
    print(f"MAKING PREDICTIONS WITH MOTIF DETECTION")
    print(f"{'='*80}")
    print(f"Number of sequences: {len(sequences)}")

    if not sequences:
        print("ERROR: No sequences to predict")
        return []

    # Get predictions with motif information
    # Use windowed prediction to handle long reads (e.g. ONT >10kb)
    try:
        predictions = classifier.predict_with_motifs_windowed(
            sequences,
            window_size=pred_cfg.get("window_size", 5000),
            step_size=pred_cfg.get("step_size", 2000),
            long_read_threshold=pred_cfg.get("long_read_threshold", 10000),
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return sequences

    # ---- Deduplication ----
    # Do NOT reclassify merged STRs — for windowed long-read predictions
    # the window-level probabilities are the correct signal. Reclassifying
    # would run the model on the full merged sequence, diluting the STR signal.
    if output_cfg.get("deduplicate", True):
        max_gap = output_cfg.get("dedup_max_gap", 50)
        min_overlap = output_cfg.get("dedup_min_overlap", 5)
        predictions = deduplicate_str_predictions(
            predictions, max_gap=max_gap, min_overlap=min_overlap,
            reclassify=False, classifier=None,
        )

    # ---- REDatlas clinical annotation ----
    if redatlas_cfg.get("enabled", True):
        try:
            from src.data.redatlas import REDatlasAnnotator
            annotator = REDatlasAnnotator(
                summary_tsv=redatlas_cfg.get("summary_tsv"),
                population_tsv=redatlas_cfg.get("population_tsv"),
                max_distance_bp=redatlas_cfg.get("max_distance_bp", 1000),
            )
            predictions = annotator.annotate_predictions(predictions)
        except Exception as e:
            logger.warning(f"REDatlas annotation skipped: {e}")

    # Count predictions
    num_str = sum(1 for s in predictions if s.get("predicted_str"))
    num_non_str = len(predictions) - num_str
    print(f"\nPredictions: {num_str} STRs, {num_non_str} non-STRs")

    str_predictions = [s for s in predictions if s.get("predicted_str")]
    sorted_strs: List[Dict] = []
    avg_str_prob = 0.0

    if str_predictions:
        avg_str_prob = sum(s["str_probability"] for s in str_predictions) / len(str_predictions)
        print(f"Average STR probability: {avg_str_prob:.4f}")

        print(f"\nTop 10 predicted STRs (highest probability):")
        sorted_strs = sorted(str_predictions, key=lambda x: x["str_probability"], reverse=True)

        for i, pred in enumerate(sorted_strs[:10], 1):
            seq_preview = pred["sequence"][:60]
            if len(pred["sequence"]) > 60:
                seq_preview += "..."
            motif = pred.get("repeat_motif", "N/A")
            count = pred.get("repeat_count", 0)
            prob = pred["str_probability"]
            disease = pred.get("redatlas_disease", "")
            disease_str = f" [{disease}]" if disease else ""
            print(f"  {i}. Prob={prob:.4f}, Motif=({motif}) x {count}{disease_str}: {seq_preview}")
    else:
        print("\nNo STR sequences predicted.")

    # ---- Save predictions ----
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2, default=str)
        print(f"\nPredictions saved to: {output_path}")

        # STRs-only file
        str_only_path = output_path.replace(".json", "_strs_only.json")
        with open(str_only_path, "w") as f:
            json.dump(str_predictions, f, indent=2, default=str)
        print(f"Predicted STRs only saved to: {str_only_path}")

        # Summary with motif + clinical statistics
        summary_path = output_path.replace(".json", "_summary.json")
        motif_counts = Counter(s.get("repeat_motif", "N/A") for s in str_predictions)
        motif_length_counts = Counter(
            len(s.get("repeat_motif", ""))
            for s in str_predictions
            if s.get("repeat_motif") != "N/A"
        )
        disease_counts = Counter(
            s.get("redatlas_disease", "none")
            for s in str_predictions
            if s.get("redatlas_match")
        )

        summary = {
            "total_sequences": len(predictions),
            "predicted_strs": num_str,
            "predicted_non_strs": num_non_str,
            "average_str_probability": float(avg_str_prob),
            "threshold": float(classifier.threshold),
            "algorithm": getattr(classifier, "algorithm", "unknown"),
            "motif_statistics": {
                "unique_motifs": len(motif_counts),
                "most_common_motifs": dict(motif_counts.most_common(10)),
                "motif_length_distribution": dict(motif_length_counts),
            },
            "clinical_annotations": {
                "disease_matches": dict(disease_counts),
                "total_disease_matches": sum(
                    1 for s in str_predictions if s.get("redatlas_match")
                ),
            },
            "top_20_str_predictions": [
                {
                    "sequence": s["sequence"][:100],
                    "probability": float(s["str_probability"]),
                    "repeat_motif": s.get("repeat_motif", "N/A"),
                    "repeat_count": s.get("repeat_count", 0),
                    "repeat_length": s.get("repeat_length", 0),
                    "canonical_fraction": s.get("canonical_fraction", 0.0),
                    "interruption_count": s.get("interruption_count", 0),
                    "chromosome": s.get("chromosome", "unknown"),
                    "position": s.get("position", -1),
                    "redatlas_disease": s.get("redatlas_disease"),
                    "redatlas_allele_class": s.get("redatlas_allele_class"),
                }
                for s in sorted_strs[:20]
            ]
            if sorted_strs
            else [],
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Prediction summary saved to: {summary_path}")

    except Exception as e:
        logger.error(f"Error saving predictions: {e}", exc_info=True)

    return predictions
