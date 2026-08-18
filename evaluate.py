"""
evaluate.py

Evaluate a trained Prompt Injection Detection model (Logistic Regression on
TF-IDF features) on held-out test data, reporting standard classification
metrics and saving diagnostic plots.

Generates:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - Confusion Matrix
    - Classification Report

Plots (saved as PNG files):
    - Confusion matrix heatmap -> confusion_matrix.png
    - ROC curve                -> roc_curve.png

Usage:
    python evaluate.py

    or as a module:
    from evaluate import run_evaluation
    metrics = run_evaluation(
        features_path="X_features.pkl",
        dataset_path="dataset_cleaned.csv",
        model_path="model.pkl",
        label_column="label",
    )
"""

import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/script use
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_model(filepath: str = "model.pkl"):
    """
    Load a trained classifier from disk.

    Args:
        filepath: Path to the pickled model.

    Returns:
        The fitted classifier.
    """
    logger.info(f"Loading model from {filepath}")
    return joblib.load(filepath)


def load_features(filepath: str = "X_features.pkl"):
    """
    Load a TF-IDF feature matrix saved by feature_extraction.py.

    Args:
        filepath: Path to the pickled feature matrix.

    Returns:
        The feature matrix (sparse or dense array).
    """
    logger.info(f"Loading TF-IDF features from {filepath}")
    X = joblib.load(filepath)
    logger.info(f"Loaded feature matrix of shape {X.shape}")
    return X


def load_labels(dataset_path: str, label_column: str) -> pd.Series:
    """
    Load labels from the cleaned dataset CSV.

    Args:
        dataset_path: Path to the cleaned dataset CSV (must contain label_column).
        label_column: Name of the column containing class labels.

    Returns:
        Series of labels, aligned by row order with the feature matrix.

    Raises:
        KeyError: If label_column is not present in the dataset.
    """
    df = pd.read_csv(dataset_path)
    if label_column not in df.columns:
        raise KeyError(
            f"Label column '{label_column}' not found in {dataset_path} "
            f"(columns: {list(df.columns)}). Evaluation requires ground-truth "
            f"labels aligned with X_features.pkl."
        )
    return df[label_column]


def load_label_encoder(filepath: str = "label_encoder.pkl"):
    """
    Load a fitted LabelEncoder from disk, if one was saved during training.

    Args:
        filepath: Path to the pickled label encoder.

    Returns:
        The fitted LabelEncoder, or None if no encoder file is found (expected
        when training labels were already numeric, e.g. 0/1).
    """
    try:
        encoder = joblib.load(filepath)
        logger.info(f"Loaded label encoder from {filepath}")
        return encoder
    except FileNotFoundError:
        logger.info(f"No label encoder found at {filepath} — assuming numeric labels")
        return None


# --------------------------------------------------------------------------- #
# Label encoding / prep (must mirror train_model.py exactly)
# --------------------------------------------------------------------------- #

def encode_labels(y: pd.Series, label_encoder=None):
    """
    Encode labels to numeric form, reusing a pre-fitted encoder if available
    so class-to-index mapping matches what the model was trained on.

    Args:
        y: Series of raw labels (numeric or string).
        label_encoder: A fitted LabelEncoder from training, or None.

    Returns:
        numpy array of encoded integer labels.
    """
    if pd.api.types.is_numeric_dtype(y):
        return y.to_numpy()
    if label_encoder is not None:
        return label_encoder.transform(y)
    # Fallback: fit a new encoder (only reached if none was saved at train time
    # and labels are non-numeric — encoding order may differ from training).
    logger.warning(
        "No saved label encoder found but labels are non-numeric — fitting a "
        "new encoder. Class index order may not match the trained model."
    )
    return LabelEncoder().fit_transform(y)


def rebuild_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Recreate the same train/test split used in train_model.py, so evaluation
    happens on the same held-out test set the model was scored on originally.

    Args:
        X: Full feature matrix.
        y: Full encoded label array.
        test_size: Fraction held out for testing (must match train_model.py).
        random_state: Random seed (must match train_model.py).

    Returns:
        Tuple of (X_test, y_test).
    """
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Rebuilt test split: {X_test.shape[0]} rows")
    return X_test, y_test


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(y_test, y_pred, y_proba) -> dict:
    """
    Compute standard classification metrics.

    Args:
        y_test: True labels (numeric).
        y_pred: Predicted labels (numeric).
        y_proba: Predicted probability of the positive class (1 = Malicious).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc.
    """
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """
    Print core evaluation metrics.

    Args:
        metrics: Dict with keys accuracy, precision, recall, f1, roc_auc.
    """
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")


def print_classification_report(y_test, y_pred, target_names=None) -> str:
    """
    Print and return the full sklearn classification report (per-class
    precision/recall/F1/support).

    Args:
        y_test: True labels.
        y_pred: Predicted labels.
        target_names: Optional list of class display names, e.g. ['Safe', 'Malicious'].

    Returns:
        The classification report as a string.
    """
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    print("\nClassification Report:")
    print(report)
    return report


def get_confusion_matrix(y_test, y_pred) -> np.ndarray:
    """
    Compute the confusion matrix.

    Args:
        y_test: True labels.
        y_pred: Predicted labels.

    Returns:
        2D numpy array confusion matrix.
    """
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    return cm


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(cm: np.ndarray, class_names: list, filepath: str = "confusion_matrix.png") -> None:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (from get_confusion_matrix).
        class_names: List of class display names in label order, e.g. ['Safe', 'Malicious'].
        filepath: Output path for the saved plot.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix — Prompt Injection Detection")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {filepath}")


def plot_roc_curve(y_test, y_proba, roc_auc: float, filepath: str = "roc_curve.png") -> None:
    """
    Plot and save the ROC curve.

    Args:
        y_test: True labels (numeric, 1 = Malicious).
        y_proba: Predicted probability of the positive class (1 = Malicious).
        roc_auc: Precomputed ROC-AUC score (shown in the legend).
        filepath: Output path for the saved plot.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random guess")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Prompt Injection Detection")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ROC curve plot to {filepath}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_evaluation(
    features_path: str = "X_features.pkl",
    dataset_path: str = "dataset_cleaned.csv",
    model_path: str = "model.pkl",
    encoder_path: str = "label_encoder.pkl",
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    confusion_matrix_path: str = "confusion_matrix.png",
    roc_curve_path: str = "roc_curve.png",
) -> dict:
    """
    Run the complete evaluation pipeline end-to-end:
    load model/features/labels -> rebuild the same test split used in
    training -> predict -> compute metrics -> print report -> save plots.

    NOTE: test_size and random_state must match the values used in
    train_model.py, or this will evaluate on a different split than the one
    the model was originally scored on (and may leak train rows into "test").

    Args:
        features_path: Path to the TF-IDF feature matrix.
        dataset_path: Path to the cleaned dataset CSV containing labels.
        model_path: Path to the trained model.
        encoder_path: Path to the label encoder (optional).
        label_column: Name of the label column in the dataset.
        test_size: Fraction held out for testing (must match training).
        random_state: Random seed (must match training).
        confusion_matrix_path: Output path for the confusion matrix plot.
        roc_curve_path: Output path for the ROC curve plot.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc,
        confusion_matrix (as a list), classification_report (as a string).
    """
    model = load_model(model_path)
    X = load_features(features_path)
    y_raw = load_labels(dataset_path, label_column)
    label_encoder = load_label_encoder(encoder_path)

    y = encode_labels(y_raw, label_encoder)
    X_test, y_test = rebuild_test_split(X, y, test_size=test_size, random_state=random_state)

    y_pred = model.predict(X_test)
    # Probability of the positive class. model.classes_ tells us which column is "1"/malicious.
    positive_class_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1
    y_proba = model.predict_proba(X_test)[:, positive_class_idx]

    # Human-readable class names in label order for report/plot labeling
    if label_encoder is not None:
        class_names = list(label_encoder.inverse_transform(sorted(np.unique(y))))
    else:
        class_names = ["Safe" if c == 0 else "Malicious" for c in sorted(np.unique(y))]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics)

    report = print_classification_report(y_test, y_pred, target_names=class_names)
    cm = get_confusion_matrix(y_test, y_pred)

    plot_confusion_matrix(cm, class_names, filepath=confusion_matrix_path)
    plot_roc_curve(y_test, y_proba, metrics["roc_auc"], filepath=roc_curve_path)

    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = report
    return metrics


if __name__ == "__main__":
    run_evaluation(
        features_path="X_features.pkl",
        dataset_path="dataset_cleaned.csv",
        model_path="model.pkl",
        encoder_path="label_encoder.pkl",
        label_column="label",
    )
