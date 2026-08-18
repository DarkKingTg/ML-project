"""
evaluation/evaluator.py

Post-training evaluation: Accuracy, Precision, Recall, F1, ROC-AUC,
Confusion Matrix, and classification report, plus saved plots.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

from utils.logger import get_logger
from utils.exceptions import BehaveGuardError

logger = get_logger(__name__)


@torch.no_grad()
def get_predictions(model, loader, device: str):
    """
    Run the model over a DataLoader and collect true labels, predicted
    labels, and predicted probabilities of the positive (malicious) class.

    Args:
        model: Trained HybridFusionClassifier (in eval mode).
        loader: DataLoader over a HybridFeatureDataset.
        device: 'cuda' or 'cpu'.

    Returns:
        Tuple of (y_true, y_pred, y_proba_positive) as numpy arrays.
    """
    model.eval()
    model.to(device)
    all_labels, all_preds, all_probs = [], [], []
    for behavioral, tfidf, embedding, labels in loader:
        behavioral, tfidf, embedding = behavioral.to(device), tfidf.to(device), embedding.to(device)
        proba = model.predict_proba(behavioral, tfidf, embedding)
        preds = torch.argmax(proba, dim=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(proba[:, 1].cpu().numpy())  # probability of class 1 (Malicious)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, class_names: list = None) -> dict:
    """
    Compute the full evaluation metric suite.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probability of the positive class.
        class_names: Optional display names for the classification report, e.g. ['Safe', 'Malicious'].

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc,
        confusion_matrix (list of lists), classification_report (str).

    Raises:
        BehaveGuardError: If metrics cannot be computed (e.g. only one class present).
    """
    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        if len(np.unique(y_true)) < 2:
            logger.warning("ROC-AUC undefined: only one class present in y_true")
            metrics["roc_auc"] = float("nan")
        else:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
        return metrics
    except Exception as e:
        raise BehaveGuardError(f"Failed to compute evaluation metrics: {e}") from e


def print_metrics(metrics: dict) -> None:
    """Print the core scalar metrics."""
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")


def plot_confusion_matrix(cm, class_names: list, filepath: str) -> None:
    """Plot and save a confusion matrix heatmap."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.asarray(cm), display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix — BehaveGuard")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {filepath}")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, roc_auc: float, filepath: str) -> None:
    """Plot and save the ROC curve."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random guess")
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — BehaveGuard")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ROC curve plot to {filepath}")
