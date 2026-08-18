"""
features/tfidf_features.py

TF-IDF feature extraction for the BehaveGuard pipeline, operating on
preprocessed (cleaned) text.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.logger import get_logger
from utils.exceptions import FeatureExtractionError, ModelLoadError

logger = get_logger(__name__)


def build_vectorizer(cfg: dict) -> TfidfVectorizer:
    """
    Construct an (unfitted) TfidfVectorizer from config.

    Args:
        cfg: Full configuration dict (reads cfg['tfidf']).

    Returns:
        An unfitted TfidfVectorizer instance.
    """
    tfidf_cfg = cfg["tfidf"]
    return TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        stop_words=tfidf_cfg["stop_words"],
    )


def fit_transform(vectorizer: TfidfVectorizer, texts):
    """
    Fit the vectorizer on cleaned texts and transform them into a dense
    TF-IDF feature matrix.

    Args:
        vectorizer: An unfitted TfidfVectorizer.
        texts: Iterable of cleaned text strings.

    Returns:
        Dense numpy array of shape (n_samples, vocab_size), dtype float32.

    Raises:
        FeatureExtractionError: If fitting/transforming fails (e.g. empty vocabulary).
    """
    texts = list(texts)
    try:
        X = vectorizer.fit_transform([str(t) for t in texts])
    except ValueError as e:
        raise FeatureExtractionError(
            f"TF-IDF fitting failed, often caused by an empty vocabulary "
            f"(e.g. all texts became empty after cleaning): {e}"
        ) from e
    logger.info(f"Fitted TF-IDF vectorizer, matrix shape: {X.shape}")
    return X.toarray().astype("float32")


def transform(vectorizer: TfidfVectorizer, texts):
    """
    Transform texts using an already-fitted vectorizer (inference time).

    Args:
        vectorizer: A fitted TfidfVectorizer.
        texts: Iterable of cleaned text strings.

    Returns:
        Dense numpy array of shape (n_samples, vocab_size), dtype float32.

    Raises:
        FeatureExtractionError: If the vectorizer is not fitted or transform fails.
    """
    try:
        X = vectorizer.transform([str(t) for t in texts])
    except Exception as e:
        raise FeatureExtractionError(f"TF-IDF transform failed: {e}") from e
    return X.toarray().astype("float32")


def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str) -> None:
    """
    Persist a fitted vectorizer to disk.

    Args:
        vectorizer: Fitted TfidfVectorizer.
        filepath: Destination path.
    """
    joblib.dump(vectorizer, filepath)
    logger.info(f"Saved TF-IDF vectorizer to {filepath}")


def load_vectorizer(filepath: str) -> TfidfVectorizer:
    """
    Load a fitted vectorizer from disk.

    Args:
        filepath: Path to the saved vectorizer.

    Returns:
        The fitted TfidfVectorizer.

    Raises:
        ModelLoadError: If the file is missing or fails to load.
    """
    try:
        return joblib.load(filepath)
    except FileNotFoundError as e:
        raise ModelLoadError(f"TF-IDF vectorizer not found at {filepath}") from e
    except Exception as e:
        raise ModelLoadError(f"Failed to load TF-IDF vectorizer from {filepath}: {e}") from e
