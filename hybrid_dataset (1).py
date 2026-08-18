"""
models/hybrid_dataset.py

PyTorch Dataset wrapping the three fused feature branches plus labels.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.exceptions import DatasetError


class HybridFeatureDataset(Dataset):
    """Wraps behavioral, TF-IDF, and embedding features + labels as a PyTorch Dataset."""

    def __init__(self, behavioral: np.ndarray, tfidf: np.ndarray, embedding: np.ndarray, labels: np.ndarray):
        """
        Args:
            behavioral: Array of shape (n_samples, behavioral_dim).
            tfidf: Array of shape (n_samples, tfidf_dim).
            embedding: Array of shape (n_samples, embedding_dim).
            labels: Array of shape (n_samples,) with integer class labels.

        Raises:
            DatasetError: If row counts across the arrays don't match.
        """
        n = len(labels)
        if not (behavioral.shape[0] == tfidf.shape[0] == embedding.shape[0] == n):
            raise DatasetError(
                f"Row count mismatch building HybridFeatureDataset: "
                f"behavioral={behavioral.shape[0]}, tfidf={tfidf.shape[0]}, "
                f"embedding={embedding.shape[0]}, labels={n}"
            )

        self.behavioral = torch.tensor(behavioral, dtype=torch.float32)
        self.tfidf = torch.tensor(tfidf, dtype=torch.float32)
        self.embedding = torch.tensor(embedding, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.behavioral[idx], self.tfidf[idx], self.embedding[idx], self.labels[idx]
