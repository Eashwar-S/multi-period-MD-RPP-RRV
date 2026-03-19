import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute classification metrics emphasizing recall, precision, F1, and PR-AUC.
    Returns a dict of zeroed metrics when y_true is empty.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    if len(y_true) == 0:
        return {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
                "PR_AUC": np.nan, "ROC_AUC": np.nan, "TN": 0, "FP": 0, "FN": 0, "TP": 0,
                "Note": "Empty evaluation set"}
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    
    # Safe computation for zero division
    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
    
    # Probability-based metrics (if not all same class)
    if len(np.unique(y_true)) > 1:
        metrics["PR_AUC"] = average_precision_score(y_true, y_prob)
        metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        metrics["Note"] = ""
    else:
        # Only one class in y_true — PR-AUC is undefined (not 0), mark explicitly
        metrics["PR_AUC"] = np.nan
        metrics["ROC_AUC"] = np.nan
        only_class = int(np.unique(y_true)[0])
        metrics["Note"] = f"Single-class test set (only label={only_class}); PR-AUC undefined"
        
    cm = confusion_matrix(y_true, y_pred)
    # Binary classification CM: [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        metrics["TN"], metrics["FP"], metrics["FN"], metrics["TP"] = cm.ravel()
    else:
        # Default fallback
        metrics["TN"], metrics["FP"], metrics["FN"], metrics["TP"] = 0, 0, 0, 0
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 0:
                metrics["TN"] = len(y_true)
            else:
                metrics["TP"] = len(y_true)
                
    return metrics

def plot_pr_curve(y_true, y_prob, model_name, output_path):
    """
    Saves a precision-recall curve plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, marker='.', label=f'{model_name} (PR-AUC={pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, output_path):
    """
    Saves a heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
def tune_threshold_on_validation(y_val_true, y_val_prob, metric="F1", min_recall=0.3):
    """
    Finds the optimal threshold on validation set based on F1 or PR-AUC metrics.
    Enforces a minimum recall floor to prevent threshold collapse to 'predict nothing'.
    """
    best_threshold = 0.5
    best_score = -1.0
    
    y_val_true = np.array(y_val_true)
    y_val_prob = np.array(y_val_prob)
    
    # Sweep thresholds from 0.1 to 0.9
    thresholds = np.linspace(0.1, 0.9, 81)
    
    for th in thresholds:
        metrics = compute_metrics(y_val_true, y_val_prob, threshold=th)
        # Reject if below minimum recall floor (prevents degenerate 'predict nothing' solutions)
        if metrics.get('Recall', 0) < min_recall and th > 0.15:
            continue
        score = metrics[metric]
        if score > best_score:
            best_score = score
            best_threshold = th
            
    return best_threshold
