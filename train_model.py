"""
train_model.py

Train a Logistic Regression classifier for prompt-injection / jailbreak
detection using TF-IDF features produced by feature_extraction.py.

Pipeline steps:
    1. Load TF-IDF features (X_features.pkl) and labels
    2. Train/test split (80/20, random_state=42)
    3. Train a Logistic Regression classifier
    4. Evaluate: accuracy, precision, recall, F1 score
    5. Save the trained model -> model.pkl
    6. Save the label encoder (if labels are non-numeric) -> label_encoder.pkl

Usage:
    python train_model.py

    or as a module:
    from train_model import run_training
    model, metrics = run_training(
        features_path="X_features.pkl",
        dataset_path="dataset_cleaned.csv",
        label_column="label",
    )
"""

import logging

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_features(filepath: str):
    """
    Load a TF-IDF feature matrix saved by feature_extraction.py.

    Args:
        filepath: Path to the pickled feature matrix (e.g. 'X_features.pkl').

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
        KeyError: If label_column is not present in the dataset. Training
            requires labels — this dataset must have a label column (e.g.
            0/1 or 'benign'/'malicious') added before a model can be trained.
    """
    df = pd.read_csv(dataset_path)
    if label_column not in df.columns:
        raise KeyError(
            f"Label column '{label_column}' not found in {dataset_path} "
            f"(columns: {list(df.columns)}). A Logistic Regression classifier "
            f"needs labels to train on — add a label column to the dataset "
            f"(matching the row order used when X_features.pkl was generated) "
            f"before running train_model.py."
        )
    return df[label_column]


# --------------------------------------------------------------------------- #
# Label encoding
# --------------------------------------------------------------------------- #

def encode_labels(y: pd.Series):
    """
    Encode labels to numeric form if they are not already numeric.

    Args:
        y: Series of raw labels (numeric or string, e.g. 'benign'/'malicious').

    Returns:
        A tuple of (y_encoded, label_encoder):
            y_encoded: numpy array of encoded integer labels.
            label_encoder: fitted LabelEncoder, or None if labels were
                already numeric and no encoding was necessary.
    """
    if pd.api.types.is_numeric_dtype(y):
        logger.info("Labels are already numeric — no label encoding needed")
        return y.to_numpy(), None

    logger.info("Labels are non-numeric — fitting LabelEncoder")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    logger.info(f"Label classes: {list(encoder.classes_)}")
    return y_encoded, encoder


# --------------------------------------------------------------------------- #
# Train / test split
# --------------------------------------------------------------------------- #

def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split features and labels into train and test sets.

    Args:
        X: Feature matrix.
        y: Encoded labels.
        test_size: Fraction of data to hold out for testing (default 0.2 = 20%).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Split data: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_logistic_regression(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        random_state: Random seed for reproducibility.

    Returns:
        The fitted LogisticRegression model.
    """
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Training complete")
    return model


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_model(model: LogisticRegression, X_test, y_test) -> dict:
    """
    Evaluate a trained model on held-out test data.

    Args:
        model: Fitted classifier.
        X_test: Test feature matrix.
        y_test: True test labels.

    Returns:
        Dict with keys: accuracy, precision, recall, f1 (all floats).
        Precision/recall/F1 use 'weighted' averaging to handle any class
        imbalance (e.g. more benign than malicious prompts, or vice versa).
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """
    Print evaluation metrics in a readable format.

    Args:
        metrics: Dict with keys accuracy, precision, recall, f1.
    """
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #

def save_model(model: LogisticRegression, filepath: str) -> None:
    """
    Persist a trained model to disk with joblib.

    Args:
        model: Fitted classifier.
        filepath: Destination path (e.g. 'model.pkl').
    """
    joblib.dump(model, filepath)
    logger.info(f"Saved trained model to {filepath}")


def save_label_encoder(encoder, filepath: str) -> None:
    """
    Persist a fitted LabelEncoder to disk with joblib, if one was used.

    Args:
        encoder: Fitted LabelEncoder, or None if labels were already numeric.
        filepath: Destination path (e.g. 'label_encoder.pkl').
    """
    if encoder is None:
        logger.info("No label encoder was used (labels were already numeric) — skipping save")
        return
    joblib.dump(encoder, filepath)
    logger.info(f"Saved label encoder to {filepath}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_training(
    features_path: str = "X_features.pkl",
    dataset_path: str = "dataset_cleaned.csv",
    label_column: str = "label",
    model_path: str = "model.pkl",
    encoder_path: str = "label_encoder.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Run the complete model training pipeline end-to-end:
    load features/labels -> encode labels -> split -> train ->
    evaluate -> print metrics -> save model (and label encoder if needed).

    Args:
        features_path: Path to the TF-IDF feature matrix (X_features.pkl).
        dataset_path: Path to the cleaned dataset CSV containing labels.
        label_column: Name of the label column in the dataset.
        model_path: Output path for the trained model.
        encoder_path: Output path for the label encoder (only saved if used).
        test_size: Fraction of data held out for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (model, metrics):
            model: the fitted LogisticRegression classifier.
            metrics: dict with accuracy, precision, recall, f1.
    """
    X = load_features(features_path)
    y_raw = load_labels(dataset_path, label_column)

    y, label_encoder = encode_labels(y_raw)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    model = train_logistic_regression(X_train, y_train, random_state=random_state)

    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)

    save_model(model, model_path)
    save_label_encoder(label_encoder, encoder_path)

    return model, metrics


if __name__ == "__main__":
    run_training(
        features_path="X_features.pkl",
        dataset_path="dataset_cleaned.csv",
        label_column="label",
        model_path="model.pkl",
        encoder_path="label_encoder.pkl",
    )
