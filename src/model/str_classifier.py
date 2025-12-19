import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from typing import List, Dict, Any
import re
from collections import Counter

class STR_Classifier:
    """Random Forest classifier that predicts whether a genetic sequence is a Short Tandem Repeat (STR)."""

    def __init__(self, threshold=0.5):
        """Initialize the STR classifier
        
        Args:
            threshold: Probability threshold for converting predictions to binary classes
                      (default: 0.5 - sequences with >= 0.5 probability are classified as STRs)
        """
        self.model = None  # Random Forest model (not trained yet)
        self.scaler = StandardScaler()  # Scaler for numerical features
        self.feature_names = []  # List of feature column names
        self.threshold = threshold  # Threshold for binary classification
        self.X = None
        self.y = None

    def _detect_tandem_repeats(self, sequence: str, min_unit_length=1, max_unit_length=6) -> Dict[str, Any]:
        """Detect tandem repeats in a DNA sequence.
        
        Args:
            sequence: DNA sequence string
            min_unit_length: Minimum repeat unit length to consider
            max_unit_length: Maximum repeat unit length to consider
            
        Returns:
            Dictionary with repeat characteristics
        """
        if not sequence or len(sequence) < 2:
            return {
                'has_repeat': False,
                'max_repeat_count': 0,
                'max_repeat_length': 0,
                'repeat_unit_length': 0,
                'repeat_purity': 0.0,
                'repeat_unit': '',
                'total_repeat_coverage': 0.0
            }
        
        sequence = sequence.upper()
        best_repeat = {
            'count': 0,
            'length': 0,
            'unit_length': 0,
            'unit': '',
            'purity': 0.0
        }
        
        # Try different repeat unit lengths
        for unit_len in range(min_unit_length, min(max_unit_length + 1, len(sequence) // 2 + 1)):
            for start_pos in range(len(sequence) - unit_len + 1):
                repeat_unit = sequence[start_pos:start_pos + unit_len]
                
                # Count consecutive repeats
                count = 0
                pos = start_pos
                while pos + unit_len <= len(sequence):
                    if sequence[pos:pos + unit_len] == repeat_unit:
                        count += 1
                        pos += unit_len
                    else:
                        break
                
                # Update best repeat if this is better
                total_length = count * unit_len
                if count >= 2 and total_length > best_repeat['length']:
                    # Calculate purity (perfect repeats / total length in repeat region)
                    purity = count / (total_length / unit_len) if total_length > 0 else 0
                    best_repeat = {
                        'count': count,
                        'length': total_length,
                        'unit_length': unit_len,
                        'unit': repeat_unit,
                        'purity': purity
                    }
        
        # Calculate coverage (what fraction of sequence is in the repeat)
        coverage = best_repeat['length'] / len(sequence) if len(sequence) > 0 else 0
        
        return {
            'has_repeat': best_repeat['count'] >= 2,
            'max_repeat_count': best_repeat['count'],
            'max_repeat_length': best_repeat['length'],
            'repeat_unit_length': best_repeat['unit_length'],
            'repeat_purity': best_repeat['purity'],
            'repeat_unit': best_repeat['unit'],
            'total_repeat_coverage': coverage
        }

    def _calculate_sequence_features(self, sequence: str) -> Dict[str, float]:
        """Calculate sequence composition and complexity features.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Dictionary of sequence features
        """
        if not sequence:
            return {
                'gc_content': 0.0,
                'sequence_length': 0,
                'entropy': 0.0,
                'homopolymer_max': 0,
                'dinucleotide_repeats': 0,
                'a_content': 0.0,
                't_content': 0.0,
                'g_content': 0.0,
                'c_content': 0.0
            }
        
        sequence = sequence.upper()
        length = len(sequence)
        
        # Nucleotide composition
        base_counts = Counter(sequence)
        a_count = base_counts.get('A', 0)
        t_count = base_counts.get('T', 0)
        g_count = base_counts.get('G', 0)
        c_count = base_counts.get('C', 0)
        
        # GC content
        gc_content = (g_count + c_count) / length if length > 0 else 0
        
        # Shannon entropy (sequence complexity)
        entropy = 0
        for base in ['A', 'T', 'G', 'C']:
            p = base_counts.get(base, 0) / length if length > 0 else 0
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Homopolymer runs (e.g., AAAA, TTTT)
        homopolymer_max = 0
        current_run = 1
        for i in range(1, length):
            if sequence[i] == sequence[i-1]:
                current_run += 1
                homopolymer_max = max(homopolymer_max, current_run)
            else:
                current_run = 1
        
        # Dinucleotide repeats (e.g., ATATAT)
        dinuc_repeats = 0
        if length >= 4:
            for i in range(length - 3):
                dinuc = sequence[i:i+2]
                if sequence[i+2:i+4] == dinuc:
                    dinuc_repeats += 1
        
        return {
            'gc_content': gc_content,
            'sequence_length': length,
            'entropy': entropy,
            'homopolymer_max': homopolymer_max,
            'dinucleotide_repeats': dinuc_repeats,
            'a_content': a_count / length if length > 0 else 0,
            't_content': t_count / length if length > 0 else 0,
            'g_content': g_count / length if length > 0 else 0,
            'c_content': c_count / length if length > 0 else 0
        }

    def _calculate_metrics(self, y_true, y_pred_proba, threshold=None):
        """Calculate classification metrics from probabilities."""
        if threshold is None:
            threshold = self.threshold
        
        # Convert probabilities to binary classes
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Calculate specificity (true negative rate)
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # ROC-AUC if we have both classes
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            roc_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'roc_auc': roc_auc
        }

    def prepare_features(self, sequences: List[Dict[str, Any]], for_prediction=False):
        """Convert sequence data into features for ML.
        
        Args:
            sequences: List of dictionaries with keys:
                - 'sequence': DNA sequence string
                - 'is_str': Boolean label (True if STR, False if not) - only needed for training
                Or can be ReadMetadata objects from bam_process.py
            for_prediction: If True, don't require 'is_str' field
            
        Returns:
            X: Feature matrix (numpy array)
            y: Binary target labels (numpy array) - None if for_prediction=True
            df: Processed dataframe
        """
        # Handle different input formats
        processed_sequences = []
        for seq_data in sequences:
            if isinstance(seq_data, dict):
                if 'query_sequence' in seq_data:  # ReadMetadata dict format
                    processed_sequences.append({
                        'sequence': seq_data.get('query_sequence', ''),
                        'is_str': seq_data.get('is_str', False)
                    })
                else:  # Standard dict format
                    processed_sequences.append({
                        'sequence': seq_data.get('sequence', ''),
                        'is_str': seq_data.get('is_str', False)
                    })
            else:  # Assume it's a ReadMetadata object
                processed_sequences.append({
                    'sequence': getattr(seq_data, 'query_sequence', ''),
                    'is_str': getattr(seq_data, 'is_str', False)
                })
        
        df = pd.DataFrame(processed_sequences)
        
        # Ensure we have required columns
        if 'sequence' not in df.columns:
            raise ValueError("Input data must contain 'sequence' field")
        
        if not for_prediction and 'is_str' not in df.columns:
            raise ValueError("Input data must contain 'is_str' field for training")
        
        # Extract STR-specific features
        str_features = df['sequence'].apply(self._detect_tandem_repeats)
        str_features_df = pd.DataFrame(str_features.tolist())
        
        # Extract general sequence features
        seq_features = df['sequence'].apply(self._calculate_sequence_features)
        seq_features_df = pd.DataFrame(seq_features.tolist())
        
        # Combine all features
        features_df = pd.concat([str_features_df, seq_features_df], axis=1)
        
        # Remove non-numeric columns
        feature_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Build feature matrix
        X = features_df[feature_cols].values.astype(float)
        
        # Build target vector (if not for prediction)
        if for_prediction:
            y = None
        else:
            y = df['is_str'].astype(int).values
        
        # Store feature names (only on first call during training)
        if not self.feature_names:
            self.feature_names = feature_cols
        
        if not for_prediction:
            print(f"\nExtracted {len(feature_cols)} features:")
            print(f"  STR features: {list(str_features_df.columns)}")
            print(f"  Sequence features: {list(seq_features_df.columns)}")
            print(f"\nClass distribution: STR={np.sum(y)}, Non-STR={len(y) - np.sum(y)}")
        
        return X, y, pd.concat([df, features_df], axis=1)

    def train(self, sequences: List[Dict[str, Any]], test_size=0.2, cv_folds=5, random_state=42):
        """Train Random Forest classifier to detect STRs with cross-validation."""
        
        print("\n" + "="*80)
        print("TRAINING STR CLASSIFIER (Random Forest)")
        print("="*80)

        # Prepare features and target from sequence data
        X, y, df = self.prepare_features(sequences, for_prediction=False)

        print(f"\nDataset: {len(X)} sequences, {X.shape[1]} features")
        print(f"Target: {np.sum(y)} STRs, {len(y) - np.sum(y)} non-STRs")

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\nTrain: {len(X_train)} samples (STR={np.sum(y_train)}, Non-STR={len(y_train)-np.sum(y_train)})")
        print(f"Test:  {len(X_test)} samples (STR={np.sum(y_test)}, Non-STR={len(y_test)-np.sum(y_test)})")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize Random Forest classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle class imbalance
            random_state=random_state,
            n_jobs=-1
        )

        print(f"\nPerforming {cv_folds}-Fold Cross-Validation...")
        
        # Cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_metrics = {
            'accuracy': [], 
            'precision': [], 
            'recall': [], 
            'f1_score': [],
            'specificity': [], 
            'roc_auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train model on this fold
            fold_model = RandomForestClassifier(
                n_estimators=100, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, class_weight='balanced',
                random_state=random_state, n_jobs=-1
            )
            fold_model.fit(X_fold_train, y_fold_train)
            
            # Get predictions
            y_val_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_fold_val, y_val_pred_proba)
            for key, value in metrics.items():
                cv_metrics[key].append(value)

        # Print CV results
        print(f"\n{cv_folds}-Fold Cross-Validation Results:")
        for metric, values in cv_metrics.items():
            print(f"  {metric.capitalize().replace('_', ' '):15s}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

        # Train final model on entire training set
        print(f"\nTraining final model on entire training set...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on hold-out test set
        print(f"\nHold-out Test Set Evaluation:")
        
        y_test_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        test_metrics = self._calculate_metrics(y_test, y_test_pred_proba)
        
        for metric, value in test_metrics.items():
            print(f"  {metric.capitalize().replace('_', ' '):15s}: {value:.4f}")

        # Print top features
        print(f"\nTop 10 Most Important Features:")
        feature_importance = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:10]
        for feature, importance in feature_importance:
            print(f"  {feature:30s}: {importance:.4f}")

        # Return comprehensive results
        return {
            **test_metrics,
            'cv_metrics': cv_metrics,
            'y_test': y_test,
            'y_test_pred_proba': y_test_pred_proba,
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
            'cv_folds': cv_folds
        }

    def predict(self, sequences: List[Dict[str, Any]]) -> np.ndarray:
        """Predict binary STR classification for input sequences.
        
        Returns:
            Binary predictions (1=STR, 0=non-STR)
        """
        proba = self.predict_proba(sequences)
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, sequences: List[Dict[str, Any]]) -> np.ndarray:
        """Return predicted STR probabilities for input sequences.
        
        Returns:
            Probability of being an STR (0.0 to 1.0)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Prepare features from input sequences (no labels needed)
        X, _, _ = self.prepare_features(sequences, for_prediction=True)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probability of positive class (STR)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_with_motifs(self, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict STR classification and extract repeat motifs for input sequences.
        
        Args:
            sequences: List of sequence dictionaries
            
        Returns:
            List of predictions with repeat motif information
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get standard predictions
        predictions = self.predict(sequences)
        probabilities = self.predict_proba(sequences)
        
        # Add repeat motif information for each sequence
        results = []
        for i, seq_dict in enumerate(sequences):
            sequence = seq_dict.get('sequence', '')
            
            # Detect repeats
            repeat_info = self._detect_tandem_repeats(sequence)
            
            result = {
                **seq_dict,
                'predicted_str': bool(predictions[i]),
                'str_probability': float(probabilities[i]),
                'repeat_motif': repeat_info.get('repeat_unit', 'N/A'),
                'repeat_count': int(repeat_info.get('max_repeat_count', 0)),
                'repeat_length': int(repeat_info.get('max_repeat_length', 0)),
                'repeat_purity': float(repeat_info.get('repeat_purity', 0.0)),
                'repeat_coverage': float(repeat_info.get('total_repeat_coverage', 0.0)),
                'has_repeat': bool(repeat_info.get('has_repeat', False))
            }
            
            results.append(result)
        
        return results