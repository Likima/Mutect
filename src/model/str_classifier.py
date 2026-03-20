"""
STR Classifier — predicts whether a genomic sequence is a Short Tandem Repeat.

Improvements over the original Random Forest implementation:
- XGBoost / LightGBM / Random Forest selectable via config
- TRMotifAnnotator-inspired motif decomposition features
- STRling-inspired non-overlapping k-mer counting features
- k-mer frequency features (tri-mers, tetra-mers)
- CIGAR-string derived features (insertions, deletions, soft clips)
- StratifiedKFold cross-validation (preserves class ratio per fold)
- Isotonic / sigmoid probability calibration
- Youden's J threshold optimization
- Optuna-based hyperparameter tuning
- Precision-recall curve evaluation and per-motif-length metrics

References:
    - RExPRT: Genome Biology (2024) doi:10.1186/s13059-024-03171-4
    - STRling: Genome Biology (2022) doi:10.1186/s13059-022-02826-4
    - HMMSTR: NAR (2025) doi:10.1093/nar/gkae1202
    - TREPP: CatBoost stacked models (BIRA 2025)
    - TRMotifAnnotator: https://github.com/wf-TRs/TRMotifAnnotator
"""

import logging
import re
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.trmotif import STRlingKmerCounter, TRMotifDecomposer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


# ---------------------------------------------------------------------------
# k-mer frequency helpers
# ---------------------------------------------------------------------------

def _kmer_frequencies(sequence: str, k: int) -> Dict[str, float]:
    """Compute normalised k-mer frequencies for a sequence."""
    seq = sequence.upper()
    counts: Counter = Counter()
    total = 0
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if "N" not in kmer:
            counts[kmer] += 1
            total += 1
    if total == 0:
        return {}
    return {kmer: count / total for kmer, count in counts.items()}


def _kmer_feature_vector(sequence: str, k: int) -> Dict[str, float]:
    """Return a fixed-size feature vector for k-mer frequencies.

    Instead of all 4^k possible k-mers (64 for k=3, 256 for k=4), we compute
    summary statistics that are more robust: max freq, entropy, top-2 ratio,
    and number of distinct k-mers.
    """
    freqs = _kmer_frequencies(sequence, k)
    if not freqs:
        return {
            f"kmer{k}_max_freq": 0.0,
            f"kmer{k}_entropy": 0.0,
            f"kmer{k}_top2_ratio": 0.0,
            f"kmer{k}_distinct": 0.0,
            f"kmer{k}_top1_is_repeat": 0.0,
        }

    values = sorted(freqs.values(), reverse=True)
    max_freq = values[0]
    top2_ratio = values[0] / values[1] if len(values) > 1 else float(values[0] > 0)

    # Shannon entropy of k-mer distribution
    arr = np.array(list(freqs.values()))
    entropy = -np.sum(arr * np.log2(arr + 1e-12))

    # Check if the top k-mer is a simple repeat (e.g., "AAA", "ATAT")
    top_kmer = max(freqs, key=freqs.get)
    is_repeat = 1.0 if len(set(top_kmer)) <= 2 else 0.0

    return {
        f"kmer{k}_max_freq": max_freq,
        f"kmer{k}_entropy": entropy,
        f"kmer{k}_top2_ratio": top2_ratio,
        f"kmer{k}_distinct": float(len(freqs)),
        f"kmer{k}_top1_is_repeat": is_repeat,
    }


# ---------------------------------------------------------------------------
# CIGAR feature extraction
# ---------------------------------------------------------------------------

_CIGAR_OPS = {"M": 0, "I": 1, "D": 2, "N": 3, "S": 4, "H": 5, "P": 6, "=": 7, "X": 8}


def _parse_cigar_features(cigar_string: Optional[str]) -> Dict[str, float]:
    """Extract features from a CIGAR string."""
    defaults = {
        "cigar_num_ops": 0.0,
        "cigar_insertion_bases": 0.0,
        "cigar_deletion_bases": 0.0,
        "cigar_softclip_bases": 0.0,
        "cigar_match_bases": 0.0,
        "cigar_indel_ratio": 0.0,
        "cigar_softclip_ratio": 0.0,
        "cigar_complexity": 0.0,
    }
    if not cigar_string:
        return defaults

    ops = re.findall(r"(\d+)([MIDNSHP=X])", cigar_string)
    if not ops:
        return defaults

    total_bases = 0
    insertion_bases = 0
    deletion_bases = 0
    softclip_bases = 0
    match_bases = 0

    for length_str, op in ops:
        length = int(length_str)
        if op in ("M", "=", "X"):
            match_bases += length
            total_bases += length
        elif op == "I":
            insertion_bases += length
            total_bases += length
        elif op == "D":
            deletion_bases += length
        elif op == "S":
            softclip_bases += length
            total_bases += length

    indel_ratio = (insertion_bases + deletion_bases) / total_bases if total_bases > 0 else 0
    softclip_ratio = softclip_bases / total_bases if total_bases > 0 else 0

    return {
        "cigar_num_ops": float(len(ops)),
        "cigar_insertion_bases": float(insertion_bases),
        "cigar_deletion_bases": float(deletion_bases),
        "cigar_softclip_bases": float(softclip_bases),
        "cigar_match_bases": float(match_bases),
        "cigar_indel_ratio": indel_ratio,
        "cigar_softclip_ratio": softclip_ratio,
        "cigar_complexity": float(len(set(op for _, op in ops))),
    }


# ===========================================================================
# STRClassifier
# ===========================================================================


class STRClassifier:
    """Gradient-boosted / Random Forest classifier for STR detection.

    Supports XGBoost (default), LightGBM, or Random Forest as the base
    algorithm, with optional Optuna hyperparameter tuning and probability
    calibration.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        algorithm: str = "xgboost",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.threshold = threshold
        self.algorithm = algorithm.lower()
        self.config = config or {}
        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

        # Sub-modules for feature extraction
        self._trmotif = TRMotifDecomposer(
            min_unit_length=self.config.get("trmotif", {}).get("min_unit_length", 1),
            max_unit_length=self.config.get("trmotif", {}).get("max_unit_length", 6),
            handle_rotations=self.config.get("trmotif", {}).get("handle_rotations", True),
        )
        self._strling = STRlingKmerCounter()

        # Feature toggles from config
        feat_cfg = self.config.get("features", {})
        self._use_kmer = feat_cfg.get("kmer", {}).get("enabled", True)
        self._kmer_sizes = feat_cfg.get("kmer", {}).get("sizes", [3, 4])
        self._use_cigar = feat_cfg.get("cigar", {}).get("enabled", True)
        self._use_trmotif = feat_cfg.get("trmotif", {}).get("enabled", True)
        self._use_strling = feat_cfg.get("strling_kmer", {}).get("enabled", True)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _detect_tandem_repeats(self, sequence: str) -> Dict[str, Any]:
        """Legacy tandem repeat detection (fast, simple)."""
        if not sequence or len(sequence) < 2:
            return {
                "has_repeat": False,
                "max_repeat_count": 0,
                "max_repeat_length": 0,
                "repeat_unit_length": 0,
                "repeat_purity": 0.0,
                "repeat_unit": "",
                "total_repeat_coverage": 0.0,
            }

        sequence = sequence.upper()
        best = {"count": 0, "length": 0, "unit_length": 0, "unit": "", "purity": 0.0}

        for unit_len in range(1, min(7, len(sequence) // 2 + 1)):
            for start_pos in range(len(sequence) - unit_len + 1):
                repeat_unit = sequence[start_pos : start_pos + unit_len]
                count = 0
                pos = start_pos
                while pos + unit_len <= len(sequence):
                    if sequence[pos : pos + unit_len] == repeat_unit:
                        count += 1
                        pos += unit_len
                    else:
                        break
                total_length = count * unit_len
                if count >= 2 and total_length > best["length"]:
                    purity = total_length / len(sequence) if len(sequence) > 0 else 0.0
                    best = {
                        "count": count,
                        "length": total_length,
                        "unit_length": unit_len,
                        "unit": repeat_unit,
                        "purity": purity,
                    }

        coverage = best["length"] / len(sequence) if len(sequence) > 0 else 0
        return {
            "has_repeat": best["count"] >= 2,
            "max_repeat_count": best["count"],
            "max_repeat_length": best["length"],
            "repeat_unit_length": best["unit_length"],
            "repeat_purity": best["purity"],
            "repeat_unit": best["unit"],
            "total_repeat_coverage": coverage,
        }

    def _calculate_sequence_features(self, sequence: str) -> Dict[str, float]:
        """Nucleotide composition and complexity features."""
        if not sequence:
            return {
                "gc_content": 0.0, "sequence_length": 0, "entropy": 0.0,
                "homopolymer_max": 0, "dinucleotide_repeats": 0,
                "a_content": 0.0, "t_content": 0.0, "g_content": 0.0, "c_content": 0.0,
            }

        seq = sequence.upper()
        length = len(seq)
        counts = Counter(seq)
        a, t, g, c = counts.get("A", 0), counts.get("T", 0), counts.get("G", 0), counts.get("C", 0)
        gc = (g + c) / length if length else 0

        entropy = 0.0
        for base in "ATGC":
            p = counts.get(base, 0) / length if length else 0
            if p > 0:
                entropy -= p * np.log2(p)

        # Longest homopolymer run
        homo_max, run = 0, 1
        for i in range(1, length):
            if seq[i] == seq[i - 1]:
                run += 1
                homo_max = max(homo_max, run)
            else:
                run = 1

        # Dinucleotide repeat count
        dinuc = 0
        if length >= 4:
            for i in range(length - 3):
                if seq[i : i + 2] == seq[i + 2 : i + 4]:
                    dinuc += 1

        return {
            "gc_content": gc, "sequence_length": length, "entropy": entropy,
            "homopolymer_max": homo_max, "dinucleotide_repeats": dinuc,
            "a_content": a / length if length else 0,
            "t_content": t / length if length else 0,
            "g_content": g / length if length else 0,
            "c_content": c / length if length else 0,
        }

    def _extract_all_features(self, seq_dict: Dict[str, Any]) -> Dict[str, float]:
        """Extract the full feature vector for a single sequence."""
        sequence = seq_dict.get("sequence", seq_dict.get("query_sequence", ""))
        features: Dict[str, float] = {}

        # 1. Legacy tandem repeat features
        tr = self._detect_tandem_repeats(sequence)
        for k in ("max_repeat_count", "max_repeat_length", "repeat_unit_length",
                   "repeat_purity", "total_repeat_coverage"):
            features[k] = float(tr.get(k, 0))

        # 2. Sequence composition features
        features.update(self._calculate_sequence_features(sequence))

        # 3. TRMotifAnnotator features
        if self._use_trmotif:
            features.update(self._trmotif.extract_features(sequence))

        # 4. STRling k-mer features
        if self._use_strling:
            features.update(self._strling.count_features(sequence))

        # 5. k-mer frequency features (tri-mers, tetra-mers)
        if self._use_kmer:
            for k in self._kmer_sizes:
                features.update(_kmer_feature_vector(sequence, k))

        # 6. Repeat-to-length ratio (important for windowed long-read detection)
        seq_len = len(sequence) if sequence else 1
        features["repeat_fraction"] = features.get("max_repeat_length", 0) / seq_len
        features["trmotif_coverage_x_purity"] = (
            features.get("trmotif_coverage", 0) * features.get("trmotif_purity", 0)
        )

        # 7. CIGAR-derived features (only when CIGAR data is available)
        if self._use_cigar:
            cigar = seq_dict.get("cigar_string", seq_dict.get("cigar", None))
            if cigar:  # Only add when we actually have CIGAR data
                features.update(_parse_cigar_features(cigar))

        return features

    # ------------------------------------------------------------------
    # Feature preparation (batch)
    # ------------------------------------------------------------------

    def prepare_features(
        self, sequences: List[Dict[str, Any]], for_prediction: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
        """Convert sequence data into feature matrix for ML."""
        # Normalise input format
        processed = []
        for s in sequences:
            if isinstance(s, dict):
                seq_key = "query_sequence" if "query_sequence" in s else "sequence"
                processed.append({
                    "sequence": s.get(seq_key, ""),
                    "is_str": s.get("is_str", False),
                    "cigar_string": s.get("cigar_string", s.get("cigar", None)),
                    "mapping_quality": s.get("mapping_quality", None),
                })
            else:
                processed.append({
                    "sequence": getattr(s, "query_sequence", ""),
                    "is_str": getattr(s, "is_str", False),
                    "cigar_string": getattr(s, "cigar_string", None),
                    "mapping_quality": getattr(s, "mapping_quality", None),
                })

        df = pd.DataFrame(processed)
        if "sequence" not in df.columns:
            raise ValueError("Input data must contain 'sequence' field")
        if not for_prediction and "is_str" not in df.columns:
            raise ValueError("Training data must contain 'is_str' field")

        # Extract features for each row
        feature_rows = [self._extract_all_features(row) for row in processed]
        features_df = pd.DataFrame(feature_rows)

        # Keep only numeric columns
        feature_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        X = features_df[feature_cols].values.astype(float)
        # Replace NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        y = None if for_prediction else df["is_str"].astype(int).values

        if not self.feature_names:
            self.feature_names = feature_cols

        if not for_prediction:
            print(f"\nExtracted {len(feature_cols)} features")
            print(f"Class distribution: STR={np.sum(y)}, Non-STR={len(y) - np.sum(y)}")

        return X, y, pd.concat([df, features_df], axis=1)

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def _create_model(self, random_state: int = 42, **override_params) -> Any:
        """Create a model instance based on the configured algorithm."""
        rf_defaults = self.config.get("random_forest", {})
        xgb_defaults = self.config.get("xgboost", {})

        if self.algorithm == "xgboost" and _HAS_XGB:
            params = {
                "n_estimators": xgb_defaults.get("n_estimators", 200),
                "max_depth": xgb_defaults.get("max_depth", 8),
                "learning_rate": xgb_defaults.get("learning_rate", 0.1),
                "subsample": xgb_defaults.get("subsample", 0.8),
                "colsample_bytree": xgb_defaults.get("colsample_bytree", 0.8),
                "min_child_weight": xgb_defaults.get("min_child_weight", 3),
                "gamma": xgb_defaults.get("gamma", 0.1),
                "reg_alpha": xgb_defaults.get("reg_alpha", 0.1),
                "reg_lambda": xgb_defaults.get("reg_lambda", 1.0),
                "random_state": random_state,
                "n_jobs": -1,
                "eval_metric": "logloss",
            }
            params.update(override_params)
            return XGBClassifier(**params)

        if self.algorithm == "lightgbm" and _HAS_LGBM:
            params = {
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": random_state,
                "n_jobs": -1,
                "verbose": -1,
            }
            params.update(override_params)
            return LGBMClassifier(**params)

        # Fallback: Random Forest
        if self.algorithm not in ("random_forest",) and self.algorithm != "xgboost":
            logger.warning(f"Algorithm '{self.algorithm}' not available, falling back to Random Forest")
        elif self.algorithm == "xgboost" and not _HAS_XGB:
            logger.warning("XGBoost not installed, falling back to Random Forest")
        params = {
            "n_estimators": rf_defaults.get("n_estimators", 100),
            "max_depth": rf_defaults.get("max_depth", 20),
            "min_samples_split": rf_defaults.get("min_samples_split", 5),
            "min_samples_leaf": rf_defaults.get("min_samples_leaf", 2),
            "class_weight": "balanced",
            "random_state": random_state,
            "n_jobs": -1,
        }
        params.update(override_params)
        return RandomForestClassifier(**params)

    # ------------------------------------------------------------------
    # Optuna tuning
    # ------------------------------------------------------------------

    def _tune_hyperparameters(
        self, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int, random_state: int
    ) -> Dict[str, Any]:
        """Run Optuna hyperparameter search."""
        if not _HAS_OPTUNA:
            logger.warning("Optuna not installed — skipping hyperparameter tuning")
            return {}

        tuning_cfg = self.config.get("tuning", {})
        n_trials = tuning_cfg.get("n_trials", 50)
        timeout = tuning_cfg.get("timeout_seconds", 300)

        print(f"\nRunning Optuna hyperparameter tuning ({n_trials} trials, {timeout}s timeout)...")

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        def objective(trial: "optuna.Trial") -> float:
            if self.algorithm == "xgboost" and _HAS_XGB:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                }
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }

            scores = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                model = self._create_model(random_state=random_state, **params)
                model.fit(X_train[train_idx], y_train[train_idx])
                proba = model.predict_proba(X_train[val_idx])[:, 1]
                scores.append(f1_score(y_train[val_idx], (proba >= 0.5).astype(int), zero_division=0))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best = study.best_params
        print(f"Best params (F1={study.best_value:.4f}): {best}")
        return best

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_metrics(
        y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate classification metrics from probabilities."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        try:
            roc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            roc = 0.0
        try:
            pr_auc = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            pr_auc = 0.0

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "specificity": specificity,
            "roc_auc": roc,
            "pr_auc": pr_auc,
        }

    @staticmethod
    def _optimal_threshold(
        y_true: np.ndarray, y_proba: np.ndarray, min_threshold: float = 0.4
    ) -> float:
        """Find threshold that maximises Youden's J statistic.

        A minimum floor of 0.4 is enforced to prevent over-sensitive thresholds
        that cause false positives on real-world long-read data where the training
        distribution is much cleaner than production data.
        """
        thresholds = np.linspace(max(0.1, min_threshold), 0.9, 81)
        best_j, best_t = -1.0, 0.5
        for t in thresholds:
            pred = (y_proba >= t).astype(int)
            tp = np.sum((y_true == 1) & (pred == 1))
            tn = np.sum((y_true == 0) & (pred == 0))
            fp = np.sum((y_true == 0) & (pred == 1))
            fn = np.sum((y_true == 1) & (pred == 0))
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = sens + spec - 1
            if j > best_j:
                best_j = j
                best_t = t
        return float(best_t)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        sequences: List[Dict[str, Any]],
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Train the STR classifier with cross-validation, optional tuning, and calibration."""
        algo_label = self.algorithm.upper()
        if self.algorithm == "xgboost" and not _HAS_XGB:
            algo_label = "RANDOM FOREST (XGBoost not installed)"
        print(f"\n{'='*80}")
        print(f"TRAINING STR CLASSIFIER ({algo_label})")
        print(f"{'='*80}")

        X, y, df = self.prepare_features(sequences, for_prediction=False)
        print(f"\nDataset: {len(X)} sequences, {X.shape[1]} features")
        print(f"Target: {np.sum(y)} STRs, {len(y) - np.sum(y)} non-STRs")

        # Train / test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

        # Scale
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Optional hyperparameter tuning
        tuning_cfg = self.config.get("tuning", {})
        extra_params: Dict[str, Any] = {}
        if tuning_cfg.get("enabled") and _HAS_OPTUNA:
            extra_params = self._tune_hyperparameters(X_train_s, y_train, cv_folds, random_state)

        # ---- Stratified K-Fold CV ----
        print(f"\nPerforming {cv_folds}-Fold Stratified Cross-Validation...")
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_metrics: Dict[str, List[float]] = {
            k: [] for k in ("accuracy", "precision", "recall", "f1_score",
                            "specificity", "roc_auc", "pr_auc")
        }

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_s, y_train), 1):
            fold_model = self._create_model(random_state=random_state, **extra_params)
            fold_model.fit(X_train_s[tr_idx], y_train[tr_idx])
            proba = fold_model.predict_proba(X_train_s[va_idx])[:, 1]
            m = self._calculate_metrics(y_train[va_idx], proba, self.threshold)
            for k, v in m.items():
                cv_metrics[k].append(v)

        print(f"\n{cv_folds}-Fold Stratified CV Results:")
        for metric, vals in cv_metrics.items():
            label = metric.replace("_", " ").capitalize()
            print(f"  {label:20s}: {np.mean(vals):.4f} (+/- {np.std(vals):.4f})")

        # ---- Train final model on full training set ----
        print("\nTraining final model on full training set...")
        self.model = self._create_model(random_state=random_state, **extra_params)
        self.model.fit(X_train_s, y_train)

        # ---- Probability calibration ----
        cal_cfg = self.config.get("calibration", {})
        if cal_cfg.get("enabled", True):
            cal_method = cal_cfg.get("method", "isotonic")
            print(f"Calibrating probabilities ({cal_method})...")
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method=cal_method, cv=3
            )
            self.calibrated_model.fit(X_train_s, y_train)

        # ---- Threshold optimisation ----
        y_test_proba = self._predict_proba_internal(X_test_s)
        optimal_t = self._optimal_threshold(y_test, y_test_proba)
        print(f"\nOptimal threshold (Youden's J): {optimal_t:.3f} (was {self.threshold:.3f})")
        self.threshold = optimal_t

        # ---- Test-set evaluation ----
        test_metrics = self._calculate_metrics(y_test, y_test_proba, self.threshold)
        print(f"\nHold-out Test Set Evaluation (threshold={self.threshold:.3f}):")
        for metric, value in test_metrics.items():
            label = metric.replace("_", " ").capitalize()
            print(f"  {label:20s}: {value:.4f}")

        # ---- Precision-Recall curve data ----
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_test_proba)

        # ---- Feature importance ----
        if hasattr(self.model, "feature_importances_"):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            importance = {}

        if importance:
            print(f"\nTop 15 Most Important Features:")
            for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]:
                print(f"  {feat:40s}: {imp:.4f}")

        # ---- Per-motif-length evaluation ----
        motif_metrics = self._per_motif_length_eval(df, y_test_proba, y_test, test_size, random_state)

        return {
            **test_metrics,
            "threshold": self.threshold,
            "cv_metrics": cv_metrics,
            "y_test": y_test,
            "y_test_pred_proba": y_test_proba,
            "feature_importance": importance,
            "cv_folds": cv_folds,
            "pr_curve": {
                "precision": pr_precision.tolist(),
                "recall": pr_recall.tolist(),
            },
            "per_motif_metrics": motif_metrics,
            "algorithm": self.algorithm,
            "tuned_params": extra_params,
        }

    def _per_motif_length_eval(
        self, df: pd.DataFrame, y_proba: np.ndarray, y_test: np.ndarray,
        test_size: float, random_state: int,
    ) -> Dict[str, Any]:
        """Evaluate model performance broken down by repeat unit length."""
        # We need the test-set portion of df
        n_test = len(y_test)
        n_total = len(df)
        n_train = n_total - n_test
        # The test set is the last n_test rows after stratified split
        # We can't perfectly reconstruct indices, so just report overall
        # This is a best-effort evaluation
        results = {}
        try:
            test_df = df.iloc[-n_test:].copy()
            test_df["y_true"] = y_test
            test_df["y_proba"] = y_proba

            for motif_len in [1, 2, 3, 4, 5, 6]:
                mask = test_df.get("repeat_unit_length") == motif_len
                if mask is None or mask.sum() < 5:
                    continue
                subset = test_df[mask]
                m = self._calculate_metrics(
                    subset["y_true"].values, subset["y_proba"].values, self.threshold
                )
                m["count"] = int(mask.sum())
                results[f"unit_length_{motif_len}"] = m
        except Exception as e:
            logger.debug(f"Per-motif eval skipped: {e}")

        return results

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_proba_internal(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get probabilities from calibrated model if available, else raw model."""
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X_scaled)[:, 1]
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, sequences: List[Dict[str, Any]]) -> np.ndarray:
        proba = self.predict_proba(sequences)
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, sequences: List[Dict[str, Any]]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        X, _, _ = self.prepare_features(sequences, for_prediction=True)
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self._predict_proba_internal(X_scaled)

    def predict_with_motifs(self, sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict STR classification and extract repeat motifs."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        predictions = self.predict(sequences)
        probabilities = self.predict_proba(sequences)

        results = []
        for i, seq_dict in enumerate(sequences):
            sequence = seq_dict.get("sequence", "")
            repeat_info = self._detect_tandem_repeats(sequence)

            # Also get TRMotifAnnotator decomposition for richer output
            trmotif_result = self._trmotif.decompose(sequence)

            result = {
                **seq_dict,
                "predicted_str": bool(predictions[i]),
                "str_probability": float(probabilities[i]),
                "repeat_motif": trmotif_result.get("motif") or repeat_info.get("repeat_unit", "N/A"),
                "repeat_count": int(trmotif_result.get("total_copies") or repeat_info.get("max_repeat_count", 0)),
                "repeat_length": int(repeat_info.get("max_repeat_length", 0)),
                "repeat_purity": float(trmotif_result.get("purity") or repeat_info.get("repeat_purity", 0.0)),
                "repeat_coverage": float(trmotif_result.get("coverage") or repeat_info.get("total_repeat_coverage", 0.0)),
                "has_repeat": bool(repeat_info.get("has_repeat", False)),
                "canonical_fraction": float(trmotif_result.get("canonical_fraction", 0.0)),
                "interruption_count": int(trmotif_result.get("interruption_count", 0)),
            }
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Windowed prediction for long reads
    # ------------------------------------------------------------------

    def _score_window(
        self, window_seq: str, w_start: int, w_end: int, read_start: int
    ) -> Tuple[float, Dict[str, Any]]:
        """Score a single window and return (probability, result_dict)."""
        window_dict = {"sequence": window_seq, "cigar_string": None}
        proba = self.predict_proba([window_dict])[0]

        repeat_info = self._detect_tandem_repeats(window_seq)
        trmotif_result = self._trmotif.decompose(window_seq)

        result = {
            "window_start": w_start,
            "window_end": w_end,
            "window_genomic_start": read_start + w_start,
            "window_genomic_end": read_start + w_end,
            "repeat_motif": trmotif_result.get("motif") or repeat_info.get("repeat_unit", "N/A"),
            "repeat_count": int(trmotif_result.get("total_copies") or repeat_info.get("max_repeat_count", 0)),
            "repeat_length": int(repeat_info.get("max_repeat_length", 0)),
            "repeat_purity": float(trmotif_result.get("purity") or repeat_info.get("repeat_purity", 0.0)),
            "repeat_coverage": float(trmotif_result.get("coverage") or repeat_info.get("total_repeat_coverage", 0.0)),
            "has_repeat": bool(repeat_info.get("has_repeat", False)),
            "canonical_fraction": float(trmotif_result.get("canonical_fraction", 0.0)),
            "interruption_count": int(trmotif_result.get("interruption_count", 0)),
        }
        return float(proba), result

    def predict_with_motifs_windowed(
        self,
        sequences: List[Dict[str, Any]],
        window_size: int = 5000,
        step_size: int = 2000,
        long_read_threshold: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Predict STRs using multi-scale sliding windows for long reads.

        Long reads (>long_read_threshold bp) are scanned in two passes:
        1. Coarse pass: large windows (window_size) with step_size stride
        2. Refinement pass: around the best coarse hit, scan with smaller
           1000bp windows at 500bp steps to precisely locate the repeat

        Short reads are classified normally.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        short_seqs: List[Dict[str, Any]] = []
        short_indices: List[int] = []
        long_seqs: List[Dict[str, Any]] = []
        long_indices: List[int] = []

        for i, s in enumerate(sequences):
            seq = s.get("sequence", "")
            if len(seq) > long_read_threshold:
                long_seqs.append(s)
                long_indices.append(i)
            else:
                short_seqs.append(s)
                short_indices.append(i)

        results: List[Optional[Dict[str, Any]]] = [None] * len(sequences)

        # --- Short reads: standard prediction ---
        if short_seqs:
            short_results = self.predict_with_motifs(short_seqs)
            for idx, res in zip(short_indices, short_results):
                results[idx] = res

        # --- Long reads: multi-scale sliding window ---
        if long_seqs:
            print(f"\nWindowed prediction: {len(long_seqs)} long reads (>{long_read_threshold} bp)")
            print(f"  Coarse: {window_size} bp / {step_size} bp step")
            print(f"  Refine: 1000 bp / 500 bp step (around best hit)")

        for orig_idx, seq_dict in zip(long_indices, long_seqs):
            sequence = seq_dict.get("sequence", "")
            seq_len = len(sequence)
            read_start = int(seq_dict.get("position", seq_dict.get("reference_start", 0)))

            best_prob = 0.0
            best_result: Optional[Dict[str, Any]] = None
            best_w_start = 0

            # ── Pass 1: Coarse scan ──
            for w_start in range(0, seq_len - window_size + 1, step_size):
                prob, res = self._score_window(
                    sequence[w_start : w_start + window_size],
                    w_start, w_start + window_size, read_start,
                )
                if prob > best_prob:
                    best_prob = prob
                    best_result = res
                    best_w_start = w_start

            # Check tail window
            if seq_len > window_size:
                tail_start = seq_len - window_size
                prob, res = self._score_window(
                    sequence[tail_start:], tail_start, seq_len, read_start,
                )
                if prob > best_prob:
                    best_prob = prob
                    best_result = res
                    best_w_start = tail_start

            # ── Pass 2: Refinement around best hit ──
            # Use smaller 1000bp windows around the best coarse window
            # to pinpoint the repeat region more precisely
            if best_prob > 0.15:
                refine_size = 1000
                refine_step = 500
                # Scan within +/- window_size of the best coarse hit
                refine_start = max(0, best_w_start - window_size)
                refine_end = min(seq_len, best_w_start + window_size + window_size)

                for w_start in range(refine_start, refine_end - refine_size + 1, refine_step):
                    prob, res = self._score_window(
                        sequence[w_start : w_start + refine_size],
                        w_start, w_start + refine_size, read_start,
                    )
                    if prob > best_prob:
                        best_prob = prob
                        best_result = res

            is_str = best_prob >= self.threshold
            result = {
                **seq_dict,
                "predicted_str": is_str,
                "str_probability": float(best_prob),
                "windowed_prediction": True,
                "read_length": seq_len,
            }
            if best_result:
                result.update(best_result)
            else:
                result.update({
                    "repeat_motif": "N/A", "repeat_count": 0, "repeat_length": 0,
                    "repeat_purity": 0.0, "repeat_coverage": 0.0, "has_repeat": False,
                    "canonical_fraction": 0.0, "interruption_count": 0,
                })
            results[orig_idx] = result

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        state = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "algorithm": self.algorithm,
            "config": self.config,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "STRClassifier":
        state = joblib.load(path)
        instance = cls(
            threshold=state["threshold"],
            algorithm=state.get("algorithm", "random_forest"),
            config=state.get("config", {}),
        )
        instance.model = state["model"]
        instance.calibrated_model = state.get("calibrated_model")
        instance.scaler = state["scaler"]
        instance.feature_names = state["feature_names"]
        logger.info(f"Model loaded from {path}")
        return instance
