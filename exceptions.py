"""
utils/exceptions.py

Custom exception hierarchy for BehaveGuard. Using specific exception types
(rather than bare Exception/ValueError everywhere) lets callers -- including
the API layer -- catch and handle failure modes precisely, and makes error
logs immediately tell you which pipeline stage failed.
"""


class BehaveGuardError(Exception):
    """Base class for all BehaveGuard-specific exceptions."""


class DatasetError(BehaveGuardError):
    """Raised for problems loading, validating, or splitting the dataset."""


class PreprocessingError(BehaveGuardError):
    """Raised when text cleaning/preprocessing fails on the input."""


class FeatureExtractionError(BehaveGuardError):
    """Raised when behavioral, TF-IDF, or transformer feature extraction fails."""


class ModelLoadError(BehaveGuardError):
    """Raised when a trained model, vectorizer, or tokenizer fails to load."""


class TrainingError(BehaveGuardError):
    """Raised when model training fails or produces an invalid result."""


class PredictionError(BehaveGuardError):
    """Raised when inference fails on a given input."""


class ConfigError(BehaveGuardError):
    """Raised when the run configuration is missing or invalid."""
