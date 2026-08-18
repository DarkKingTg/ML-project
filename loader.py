"""
dataset/loader.py

Loads and validates the raw labeled dataset, and performs the train/test split
used consistently across training and evaluation.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import get_logger
from utils.exceptions import DatasetError

logger = get_logger(__name__)


def load_raw_dataset(csv_path: str, text_column: str, label_column: str) -> pd.DataFrame:
    """
    Load the raw labeled dataset from CSV and validate its structure.

    Args:
        csv_path: Path to the CSV file.
        text_column: Expected name of the text/prompt column.
        label_column: Expected name of the label column.

    Returns:
        DataFrame with at least [text_column, label_column], rows with
        missing values in either column dropped.

    Raises:
        DatasetError: If the file is missing, empty, or lacks the required columns.
    """
    path = Path(csv_path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {csv_path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise DatasetError(f"Dataset file is empty: {csv_path}") from e
    except Exception as e:
        raise DatasetError(f"Failed to read dataset CSV at {csv_path}: {e}") from e

    missing_cols = [c for c in (text_column, label_column) if c not in df.columns]
    if missing_cols:
        raise DatasetError(
            f"Dataset at {csv_path} is missing required column(s) {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    before = len(df)
    df = df.dropna(subset=[text_column, label_column]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with missing text/label values")

    if len(df) == 0:
        raise DatasetError(f"Dataset at {csv_path} has no valid rows after dropping missing values")

    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    return df


def split_dataset(df: pd.DataFrame, label_column: str, test_size: float, random_state: int):
    """
    Perform a stratified train/test split.

    Args:
        df: Full dataset DataFrame.
        label_column: Name of the label column (used for stratification).
        test_size: Fraction to hold out for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df), each with a reset index.

    Raises:
        DatasetError: If splitting fails (e.g. a class has too few samples to stratify).
    """
    try:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[label_column]
        )
    except ValueError as e:
        raise DatasetError(
            f"Failed to create a stratified train/test split (often caused by a "
            f"class with too few samples): {e}"
        ) from e

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    logger.info(f"Split dataset — train: {len(train_df)} rows, test: {len(test_df)} rows")
    return train_df, test_df
