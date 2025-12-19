
import json
from pathlib import Path
from typing import Any, Dict, List

from src.model.str_classifier import STR_Classifier


def train_str_classifier(training_data: List[Dict], test_size: float = 0.2, 
                         cv_folds: int = 5, threshold: float = 0.5,
                         output_dir: str = "output") -> Dict[str, Any]:
    """Train STR classifier and save results.
    
    Args:
        training_data: List of labeled sequences
        test_size: Fraction of data to use for testing
        cv_folds: Number of cross-validation folds
        threshold: Classification threshold
        output_dir: Directory to save results
        
    Returns:
        Dictionary with training results and metrics
    """
    print(f"\n{'='*80}")
    print(f"TRAINING STR CLASSIFIER")
    print(f"{'='*80}")
    
    # Initialize classifier
    classifier = STR_Classifier(threshold=threshold)
    
    # Train model
    results = classifier.train(
        sequences=training_data,
        test_size=test_size,
        cv_folds=cv_folds,
        random_state=42
    )
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results_path = Path(output_dir) / "training_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'specificity': float(results['specificity']),
            'roc_auc': float(results['roc_auc']),
            'cv_folds': results['cv_folds'],
            'cv_metrics': {
                metric: [float(v) for v in values]
                for metric, values in results['cv_metrics'].items()
            },
            'feature_importance': {
                str(k): float(v) for k, v in results['feature_importance'].items()
            }
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return {
        'classifier': classifier,
        'results': results
    }


def predict_str_sequences(classifier: STR_Classifier, sequences: List[Dict],
                          output_path: str = "output/predictions.json") -> List[Dict]:
    """Make predictions on new sequences with repeat motif detection.
    
    Args:
        classifier: Trained STR_Classifier
        sequences: List of sequences to classify
        output_path: Where to save predictions
        
    Returns:
        Sequences with predictions and motif information added
    """
    print(f"\n{'='*80}")
    print(f"MAKING PREDICTIONS WITH MOTIF DETECTION")
    print(f"{'='*80}")
    print(f"Number of sequences: {len(sequences)}")
    
    if not sequences:
        print("ERROR: No sequences to predict")
        return []
    
    # Get predictions with motif information using predict_with_motifs
    try:
        sequences_with_predictions = classifier.predict_with_motifs(sequences)
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return sequences
    
    # Count predictions
    num_str = sum(1 for s in sequences_with_predictions if s.get('predicted_str'))
    num_non_str = len(sequences_with_predictions) - num_str
    
    print(f"\nPredictions: {num_str} STRs, {num_non_str} non-STRs")
    
    # Calculate statistics
    str_predictions = [s for s in sequences_with_predictions if s.get('predicted_str')]
    sorted_strs = []  # Initialize here to avoid UnboundLocalError
    avg_str_prob = 0.0
    
    if str_predictions:
        avg_str_prob = sum(s['str_probability'] for s in str_predictions) / len(str_predictions)
        print(f"Average STR probability: {avg_str_prob:.4f}")
        
        # Show top predicted STRs with motif information
        print(f"\nTop 10 predicted STRs (highest probability):")
        sorted_strs = sorted(str_predictions, key=lambda x: x['str_probability'], reverse=True)
        
        for i, pred in enumerate(sorted_strs[:10], 1):
            seq_preview = pred['sequence'][:60]
            if len(pred['sequence']) > 60:
                seq_preview += "..."
            
            motif = pred.get('repeat_motif', 'N/A')
            count = pred.get('repeat_count', 0)
            prob = pred['str_probability']
            
            print(f"  {i}. Prob={prob:.4f}, Motif=({motif}) x {count}: {seq_preview}")
    else:
        print("\nNo STR sequences predicted.")
    
    # Save predictions
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            json.dump(sequences_with_predictions, f, indent=2, default=str)
        print(f"\nPredictions saved to: {output_path}")
        
        # Save STRs-only file with motif information
        str_only_path = output_path.replace('.json', '_strs_only.json')
        with open(str_only_path, 'w') as f:
            json.dump(str_predictions, f, indent=2, default=str)
        print(f"Predicted STRs only saved to: {str_only_path}")
        
        # Save summary with motif statistics
        summary_path = output_path.replace('.json', '_summary.json')
        
        # Count motif types
        from collections import Counter
        motif_counts = Counter(s.get('repeat_motif', 'N/A') for s in str_predictions)
        motif_length_counts = Counter(len(s.get('repeat_motif', '')) for s in str_predictions if s.get('repeat_motif') != 'N/A')
        
        summary = {
            'total_sequences': len(sequences_with_predictions),
            'predicted_strs': num_str,
            'predicted_non_strs': num_non_str,
            'average_str_probability': float(avg_str_prob),
            'threshold': float(classifier.threshold),
            'motif_statistics': {
                'unique_motifs': len(motif_counts),
                'most_common_motifs': dict(motif_counts.most_common(10)),
                'motif_length_distribution': dict(motif_length_counts)
            },
            'top_20_str_predictions': [
                {
                    'sequence': s['sequence'][:100],
                    'probability': float(s['str_probability']),
                    'repeat_motif': s.get('repeat_motif', 'N/A'),
                    'repeat_count': s.get('repeat_count', 0),
                    'repeat_length': s.get('repeat_length', 0),
                    'chromosome': s.get('chromosome', 'unknown'),
                    'position': s.get('position', -1)
                }
                for s in sorted_strs[:20]  # Now safe to use sorted_strs
            ] if sorted_strs else []  # Empty list if no STRs predicted
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Prediction summary with motif statistics saved to: {summary_path}")
        
    except Exception as e:
        print(f"ERROR saving predictions: {e}")
        import traceback
        traceback.print_exc()
    
    return sequences_with_predictions