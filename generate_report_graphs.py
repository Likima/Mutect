#!/usr/bin/env python3
"""
Generate comprehensive visualization graphs for the STR classification report.

This script creates all important visualizations based on the output data:
- Model performance metrics
- Feature importance analysis
- Data distribution comparisons
- Cross-validation results
- Feature distributions (STR vs Non-STR)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path("output")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_model_performance_metrics(results: Dict):
    """Plot 1: Model performance metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']
    values = [results[m] for m in metrics]
    
    # Bar plot of all metrics
    ax1 = axes[0, 0]
    bars = ax1.bar(metrics, values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'])
    ax1.set_ylim([0.95, 1.0])
    ax1.set_ylabel('Score')
    ax1.set_title('Test Set Performance Metrics')
    ax1.tick_params(axis='x', rotation=45)
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Cross-validation box plot
    ax2 = axes[0, 1]
    cv_data = []
    cv_labels = []
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        cv_data.extend(results['cv_metrics'][metric])
        cv_labels.extend([metric] * len(results['cv_metrics'][metric]))
    
    df_cv = pd.DataFrame({'Metric': cv_labels, 'Score': cv_data})
    sns.boxplot(data=df_cv, x='Metric', y='Score', ax=ax2, hue='Metric', palette='Set2', legend=False)
    ax2.set_title('5-Fold Cross-Validation Results')
    ax2.set_ylabel('Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim([0.95, 1.0])
    
    # Metrics comparison: Test vs CV mean
    ax3 = axes[1, 0]
    test_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    test_values = [results[m] for m in test_metrics]
    cv_means = [np.mean(results['cv_metrics'][m]) for m in test_metrics]
    cv_stds = [np.std(results['cv_metrics'][m]) for m in test_metrics]
    
    x = np.arange(len(test_metrics))
    width = 0.35
    ax3.bar(x - width/2, test_values, width, label='Test Set', color='#3498db', alpha=0.8)
    ax3.bar(x + width/2, cv_means, width, label='CV Mean', color='#e74c3c', alpha=0.8, yerr=cv_stds, capsize=5)
    ax3.set_ylabel('Score')
    ax3.set_title('Test Set vs Cross-Validation Mean')
    ax3.set_xticks(x)
    ax3.set_xticklabels(test_metrics, rotation=45)
    ax3.legend()
    ax3.set_ylim([0.95, 1.0])
    
    # ROC AUC visualization
    ax4 = axes[1, 1]
    roc_auc = results['roc_auc']
    cv_roc = results['cv_metrics']['roc_auc']
    ax4.bar(['Test Set', 'CV Mean'], 
            [roc_auc, np.mean(cv_roc)],
            color=['#2ecc71', '#3498db'], alpha=0.8)
    ax4.errorbar(['CV Mean'], [np.mean(cv_roc)], 
                 yerr=[np.std(cv_roc)], fmt='none', color='black', capsize=10)
    ax4.set_ylabel('ROC AUC Score')
    ax4.set_title(f'ROC AUC: {roc_auc:.4f}')
    ax4.set_ylim([0.99, 1.0])
    ax4.text(0, roc_auc, f'{roc_auc:.4f}', ha='center', va='bottom', fontweight='bold')
    ax4.text(1, np.mean(cv_roc), f'{np.mean(cv_roc):.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '1_model_performance_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {FIGURES_DIR / '1_model_performance_metrics.png'}")
    plt.close()


def plot_feature_importance(results: Dict):
    """Plot 2: Feature importance analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    importance = results['feature_importance']
    features = list(importance.keys())
    values = list(importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(values)[::-1]
    features_sorted = [features[i] for i in sorted_idx]
    values_sorted = [values[i] for i in sorted_idx]
    
    # Horizontal bar plot
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features_sorted)))
    bars = ax1.barh(features_sorted, values_sorted, color=colors)
    ax1.set_xlabel('Importance Score')
    ax1.set_title('Feature Importance (Random Forest)')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values_sorted)):
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left', va='center', fontsize=9)
    
    # Pie chart for top features
    ax2 = axes[1]
    top_n = 8
    top_features = features_sorted[:top_n]
    top_values = values_sorted[:top_n]
    other_value = sum(values_sorted[top_n:])
    
    if other_value > 0:
        pie_features = top_features + ['Others']
        pie_values = top_values + [other_value]
    else:
        pie_features = top_features
        pie_values = top_values
    
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_features)))
    wedges, texts, autotexts = ax2.pie(pie_values, labels=pie_features, autopct='%1.2f%%',
                                       colors=colors_pie, startangle=90)
    ax2.set_title(f'Top {top_n} Features Contribution')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '2_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {FIGURES_DIR / '2_feature_importance.png'}")
    plt.close()


def plot_data_distributions():
    """Plot 3: Data distribution analysis."""
    str_data = load_json(OUTPUT_DIR / 'str_variants.json')
    normal_data = load_json(OUTPUT_DIR / 'normal_sequences.json')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Sequence length distribution
    ax1 = axes[0, 0]
    str_lengths = [s.get('length', len(s.get('sequence', ''))) for s in str_data]
    normal_lengths = [s.get('length', len(s.get('sequence', ''))) for s in normal_data]
    
    ax1.hist(str_lengths, bins=50, alpha=0.7, label='STR', color='#e74c3c', density=True)
    ax1.hist(normal_lengths, bins=50, alpha=0.7, label='Non-STR', color='#3498db', density=True)
    ax1.set_xlabel('Sequence Length (bp)')
    ax1.set_ylabel('Density')
    ax1.set_title('Sequence Length Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Chromosome distribution
    ax2 = axes[0, 1]
    str_chrs = [s.get('chr', 'Unknown') for s in str_data]
    normal_chrs = [s.get('chr', 'Unknown') for s in normal_data]
    
    str_chr_counts = Counter(str_chrs)
    normal_chr_counts = Counter(normal_chrs)
    
    all_chrs = sorted(set(str_chr_counts.keys()) | set(normal_chr_counts.keys()))
    str_counts = [str_chr_counts.get(c, 0) for c in all_chrs]
    normal_counts = [normal_chr_counts.get(c, 0) for c in all_chrs]
    
    x = np.arange(len(all_chrs))
    width = 0.35
    ax2.bar(x - width/2, str_counts, width, label='STR', color='#e74c3c', alpha=0.8)
    ax2.bar(x + width/2, normal_counts, width, label='Non-STR', color='#3498db', alpha=0.8)
    ax2.set_xlabel('Chromosome')
    ax2.set_ylabel('Count')
    ax2.set_title('Chromosome Distribution')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_chrs, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Gene distribution (top genes)
    ax3 = axes[0, 2]
    str_genes = [s.get('gene', 'N/A') for s in str_data if s.get('gene') != 'N/A']
    gene_counts = Counter(str_genes)
    top_genes = dict(gene_counts.most_common(10))
    
    if top_genes:
        ax3.barh(list(top_genes.keys()), list(top_genes.values()), color='#2ecc71')
        ax3.set_xlabel('Count')
        ax3.set_title('Top 10 Genes with STR Variants')
        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No gene annotations', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Gene Distribution')
    
    # Length box plot comparison
    ax4 = axes[1, 0]
    length_data = pd.DataFrame({
        'Length': str_lengths + normal_lengths,
        'Type': ['STR'] * len(str_lengths) + ['Non-STR'] * len(normal_lengths)
    })
    sns.boxplot(data=length_data, x='Type', y='Length', ax=ax4, hue='Type', palette=['#e74c3c', '#3498db'], legend=False)
    ax4.set_title('Sequence Length Comparison')
    ax4.set_ylabel('Length (bp)')
    ax4.grid(axis='y', alpha=0.3)
    
    # Dataset size comparison
    ax5 = axes[1, 1]
    categories = ['STR Variants', 'Normal Sequences', 'Balanced Dataset']
    sizes = [len(str_data), len(normal_data), min(len(str_data), len(normal_data)) * 2]
    colors_bar = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax5.bar(categories, sizes, color=colors_bar, alpha=0.8)
    ax5.set_ylabel('Number of Sequences')
    ax5.set_title('Dataset Sizes')
    ax5.grid(axis='y', alpha=0.3)
    for bar, size in zip(bars, sizes):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{size:,}', ha='center', va='bottom', fontweight='bold')
    
    # Class balance visualization
    ax6 = axes[1, 2]
    balanced_data = load_json(OUTPUT_DIR / 'balanced_dataset.json')
    str_count = sum(1 for s in balanced_data if s.get('is_str'))
    non_str_count = len(balanced_data) - str_count
    
    ax6.pie([str_count, non_str_count], labels=['STR', 'Non-STR'], 
           autopct='%1.1f%%', colors=['#e74c3c', '#3498db'], startangle=90)
    ax6.set_title(f'Balanced Dataset\n({len(balanced_data)} total sequences)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '3_data_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {FIGURES_DIR / '3_data_distributions.png'}")
    plt.close()


def plot_cross_validation_results(results: Dict):
    """Plot 4: Detailed cross-validation analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cross-Validation Analysis (5-Fold)', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']
    fold_numbers = list(range(1, 6))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        cv_values = results['cv_metrics'][metric]
        test_value = results[metric]
        
        # Plot CV folds
        ax.plot(fold_numbers, cv_values, 'o-', linewidth=2, markersize=8, 
               label='CV Folds', color='#3498db', alpha=0.7)
        ax.axhline(y=np.mean(cv_values), color='#e74c3c', linestyle='--', 
                  linewidth=2, label=f'CV Mean: {np.mean(cv_values):.4f}')
        ax.axhline(y=test_value, color='#2ecc71', linestyle='--', 
                  linewidth=2, label=f'Test: {test_value:.4f}')
        ax.fill_between(fold_numbers, 
                        [np.mean(cv_values) - np.std(cv_values)] * 5,
                        [np.mean(cv_values) + np.std(cv_values)] * 5,
                        alpha=0.2, color='#3498db', label='±1 Std Dev')
        
        ax.set_xlabel('Fold Number')
        ax.set_ylabel('Score')
        ax.set_title(f'{metric.upper().replace("_", " ")}')
        ax.set_xticks(fold_numbers)
        ax.set_ylim([min(min(cv_values), test_value) - 0.01, 
                    max(max(cv_values), test_value) + 0.01])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '4_cross_validation.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {FIGURES_DIR / '4_cross_validation.png'}")
    plt.close()


def plot_feature_comparisons():
    """Plot 5: Feature distributions comparing STR vs Non-STR."""
    from src.model.str_classifier import STR_Classifier
    
    # Load data
    balanced_data = load_json(OUTPUT_DIR / 'balanced_dataset.json')
    
    # Sample for faster computation (if dataset is large)
    if len(balanced_data) > 1000:
        import random
        balanced_data = random.sample(balanced_data, 1000)
    
    # Extract features
    classifier = STR_Classifier()
    X, y, df_features = classifier.prepare_features(balanced_data, for_prediction=False)
    
    # Get feature names
    feature_cols = [col for col in df_features.columns 
                   if col not in ['sequence', 'is_str', 'uid', 'gene', 'chr', 'start', 'end', 
                                 'assembly', 'variant_type', 'title', 'clinical_significance',
                                 'review_status', 'condition', 'consequence', 'label']]
    
    # Select key features to plot
    key_features = ['entropy', 'max_repeat_length', 'gc_content', 'sequence_length',
                   'max_repeat_count', 'total_repeat_coverage', 'c_content', 'g_content',
                   'dinucleotide_repeats', 'homopolymer_max']
    
    available_features = [f for f in key_features if f in feature_cols]
    
    n_features = len(available_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    fig.suptitle('Feature Distributions: STR vs Non-STR', fontsize=16, fontweight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, feature in enumerate(available_features):
        ax = axes[idx // n_cols, idx % n_cols]
        
        str_values = df_features[df_features['is_str'] == True][feature].dropna()
        non_str_values = df_features[df_features['is_str'] == False][feature].dropna()
        
        ax.hist(str_values, bins=30, alpha=0.6, label='STR', color='#e74c3c', density=True)
        ax.hist(non_str_values, bins=30, alpha=0.6, label='Non-STR', color='#3498db', density=True)
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.set_title(f'{feature.replace("_", " ").title()}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_features, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '5_feature_comparisons.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {FIGURES_DIR / '5_feature_comparisons.png'}")
    plt.close()


def plot_roc_curve():
    """Plot ROC curve for the STR classifier."""
    from src.model.str_classifier import STR_Classifier
    from sklearn.model_selection import train_test_split
    
    print("Generating ROC curve...")
    print("  Loading balanced dataset...")
    balanced_data = load_json(OUTPUT_DIR / 'balanced_dataset.json')
    results = load_json(OUTPUT_DIR / 'training_results.json')
    
    # Initialize classifier
    classifier = STR_Classifier(threshold=0.5)
    
    # Prepare features
    print("  Preparing features...")
    X, y, df_features = classifier.prepare_features(balanced_data, for_prediction=False)
    
    # Split data the same way as during training (random_state=42)
    print("  Splitting data (matching training split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_scaled = classifier.scaler.fit_transform(X_train)
    X_test_scaled = classifier.scaler.transform(X_test)
    
    # Train model
    print("  Training model to generate predictions...")
    from sklearn.ensemble import RandomForestClassifier
    classifier.model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    classifier.model.fit(X_train_scaled, y_train)
    
    # Get predictions
    print("  Generating predictions...")
    y_test_pred_proba = classifier.model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(fpr, tpr, color='#3498db', lw=2, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.5000)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve - STR Classification', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add text annotation with key metrics
    textstr = f'Test Set Performance:\n'
    textstr += f'AUC = {roc_auc:.4f}\n'
    textstr += f'Accuracy = {results["accuracy"]:.4f}\n'
    textstr += f'Precision = {results["precision"]:.4f}\n'
    textstr += f'Recall = {results["recall"]:.4f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.6, 0.2, textstr, fontsize=10, verticalalignment='top',
            bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Save to both main figures directory and a dedicated subdirectory
    roc_dir = FIGURES_DIR / '1_model_performance_metrics'
    roc_dir.mkdir(exist_ok=True)
    
    plt.savefig(roc_dir / 'roc_auc.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {roc_dir / 'roc_auc.png'}")
    plt.close()


def get_real_predictions():
    """Helper function to get real predictions on test set.
    
    Returns:
        y_test: True labels
        y_test_pred: Predicted labels (binary)
        y_test_pred_proba: Predicted probabilities
    """
    from src.model.str_classifier import STR_Classifier
    from sklearn.model_selection import train_test_split
    
    # Load data
    balanced_data = load_json(OUTPUT_DIR / 'balanced_dataset.json')
    
    # Initialize classifier
    classifier = STR_Classifier(threshold=0.5)
    
    # Prepare features
    X, y, df_features = classifier.prepare_features(balanced_data, for_prediction=False)
    
    # Split data the same way as during training (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_scaled = classifier.scaler.fit_transform(X_train)
    X_test_scaled = classifier.scaler.transform(X_test)
    
    # Train model
    from sklearn.ensemble import RandomForestClassifier
    classifier.model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    classifier.model.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_test_pred_proba = classifier.model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    
    return y_test, y_test_pred, y_test_pred_proba


def plot_confusion_matrix():
    """Plot real confusion matrix for the STR classifier."""
    print("Generating confusion matrix...")
    
    # Get real predictions
    y_test, y_test_pred, _ = get_real_predictions()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=['Non-STR', 'STR'], yticklabels=['Non-STR', 'STR'],
               cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 16, 'fontweight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix - STR Classification', fontsize=16, fontweight='bold')
    
    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    textstr = f'Metrics:\n'
    textstr += f'Accuracy: {accuracy:.4f}\n'
    textstr += f'Precision: {precision:.4f}\n'
    textstr += f'Recall: {recall:.4f}\n'
    textstr += f'Specificity: {specificity:.4f}\n'
    textstr += f'F1 Score: {f1:.4f}\n\n'
    textstr += f'TN: {tn}, FP: {fp}\n'
    textstr += f'FN: {fn}, TP: {tp}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.15, 0.5, textstr, fontsize=11, verticalalignment='center',
            bbox=props, family='monospace', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to summary dashboard directory
    cm_dir = FIGURES_DIR / '6_summary_dashboard'
    cm_dir.mkdir(exist_ok=True)
    
    plt.savefig(cm_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {cm_dir / 'confusion_matrix.png'}")
    plt.close()
    
    return cm


def plot_summary_dashboard():
    """Plot 6: Summary dashboard with key metrics."""
    results = load_json(OUTPUT_DIR / 'training_results.json')
    balanced_data = load_json(OUTPUT_DIR / 'balanced_dataset.json')
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('STR Classification Pipeline - Summary Dashboard', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Key metrics (large)
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    values = [results['accuracy'], results['precision'], results['recall'], 
             results['f1_score'], results['roc_auc']]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylim([0.95, 1.0])
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Top features
    ax2 = fig.add_subplot(gs[0, 2:])
    importance = results['feature_importance']
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    features = [f[0].replace('_', ' ').title() for f in top_features]
    values = [f[1] for f in top_features]
    ax2.barh(features, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, 5)))
    ax2.set_xlabel('Importance', fontsize=11)
    ax2.set_title('Top 5 Most Important Features', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Dataset info
    ax3 = fig.add_subplot(gs[1, 0])
    str_count = sum(1 for s in balanced_data if s.get('is_str'))
    non_str_count = len(balanced_data) - str_count
    ax3.pie([str_count, non_str_count], labels=['STR', 'Non-STR'], 
           autopct='%1.1f%%', colors=['#e74c3c', '#3498db'], startangle=90,
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title(f'Dataset Balance\n({len(balanced_data)} sequences)', 
                 fontsize=12, fontweight='bold')
    
    # 4. CV stability
    ax4 = fig.add_subplot(gs[1, 1])
    cv_acc = results['cv_metrics']['accuracy']
    ax4.plot(range(1, 6), cv_acc, 'o-', linewidth=2, markersize=10, color='#3498db')
    ax4.axhline(y=np.mean(cv_acc), color='#e74c3c', linestyle='--', linewidth=2)
    ax4.fill_between(range(1, 6), 
                     [np.mean(cv_acc) - np.std(cv_acc)] * 5,
                     [np.mean(cv_acc) + np.std(cv_acc)] * 5,
                     alpha=0.2, color='#3498db')
    ax4.set_xlabel('Fold', fontsize=11)
    ax4.set_ylabel('Accuracy', fontsize=11)
    ax4.set_title(f'CV Stability\n(Mean: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f})',
                 fontsize=12, fontweight='bold')
    ax4.set_xticks(range(1, 6))
    ax4.grid(alpha=0.3)
    
    # 5. Confusion matrix (real, not estimated)
    ax5 = fig.add_subplot(gs[1, 2:])
    print("  Generating real confusion matrix for dashboard...")
    y_test, y_test_pred, _ = get_real_predictions()
    cm = confusion_matrix(y_test, y_test_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
               xticklabels=['Non-STR', 'STR'], yticklabels=['Non-STR', 'STR'],
               cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    ax5.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax5.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # 6. Metrics comparison
    ax6 = fig.add_subplot(gs[2, :2])
    metrics_short = ['Acc', 'Prec', 'Rec', 'F1', 'Spec', 'AUC']
    test_vals = [results['accuracy'], results['precision'], results['recall'],
                results['f1_score'], results['specificity'], results['roc_auc']]
    cv_means = [np.mean(results['cv_metrics'][m]) for m in 
               ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc']]
    
    x = np.arange(len(metrics_short))
    width = 0.35
    ax6.bar(x - width/2, test_vals, width, label='Test Set', color='#2ecc71', alpha=0.8)
    ax6.bar(x + width/2, cv_means, width, label='CV Mean', color='#3498db', alpha=0.8)
    ax6.set_ylabel('Score', fontsize=11)
    ax6.set_title('Test Set vs Cross-Validation Comparison', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics_short)
    ax6.legend(fontsize=11)
    ax6.set_ylim([0.95, 1.0])
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Feature importance (all)
    ax7 = fig.add_subplot(gs[2, 2:])
    importance = results['feature_importance']
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features = [f[0].replace('_', '\n') for f, _ in sorted_features]
    values = [v for _, v in sorted_features]
    ax7.barh(features, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(features))))
    ax7.set_xlabel('Importance Score', fontsize=11)
    ax7.set_title('All Feature Importances', fontsize=14, fontweight='bold')
    ax7.grid(axis='x', alpha=0.3)
    
    plt.savefig(FIGURES_DIR / '6_summary_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {FIGURES_DIR / '6_summary_dashboard.png'}")
    plt.close()


def main():
    """Generate all report graphs."""
    print("\n" + "="*80)
    print("GENERATING REPORT GRAPHS")
    print("="*80 + "\n")
    
    # Load training results
    results = load_json(OUTPUT_DIR / 'training_results.json')
    
    # Generate all plots
    print("Generating visualizations...\n")
    plot_model_performance_metrics(results)
    plot_feature_importance(results)
    plot_data_distributions()
    plot_cross_validation_results(results)
    plot_feature_comparisons()
    plot_roc_curve()
    plot_confusion_matrix()
    plot_summary_dashboard()
    
    print("\n" + "="*80)
    print("ALL GRAPHS GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nGraphs saved to: {FIGURES_DIR.absolute()}")
    print("\nGenerated files:")
    for i, fname in enumerate(sorted(FIGURES_DIR.glob("*.png")), 1):
        print(f"  {i}. {fname.name}")


if __name__ == "__main__":
    main()

