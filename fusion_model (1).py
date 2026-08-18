"""
models/fusion_model.py

PyTorch hybrid classifier: concatenates behavioral features, TF-IDF
features, and transformer (or hashing-fallback) embeddings into a single
fused vector, then classifies via a feed-forward network.
"""

import torch
import torch.nn as nn

from utils.logger import get_logger
from utils.exceptions import ModelLoadError

logger = get_logger(__name__)


class HybridFusionClassifier(nn.Module):
    """
    Feed-forward classifier over fused [behavioral | TF-IDF | embedding]
    features:
        concat -> (Linear -> ReLU -> Dropout) x N -> Linear (logits)
    """

    def __init__(
        self,
        behavioral_dim: int,
        tfidf_dim: int,
        embedding_dim: int,
        hidden_dims: list = None,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            behavioral_dim: Dimensionality of the behavioral feature vector.
            tfidf_dim: Dimensionality of the TF-IDF feature vector.
            embedding_dim: Dimensionality of the embedding vector (transformer
                hidden_size, or hashing embedder hidden_size).
            hidden_dims: List of hidden layer sizes. Defaults to [256, 64].
            num_classes: Number of output classes.
            dropout: Dropout probability after each hidden layer.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 64]

        self.behavioral_dim = behavioral_dim
        self.tfidf_dim = tfidf_dim
        self.embedding_dim = embedding_dim
        self.input_dim = behavioral_dim + tfidf_dim + embedding_dim

        layers = []
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, behavioral: torch.Tensor, tfidf: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            behavioral: Tensor of shape (batch, behavioral_dim).
            tfidf: Tensor of shape (batch, tfidf_dim).
            embedding: Tensor of shape (batch, embedding_dim).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        fused = torch.cat([behavioral, tfidf, embedding], dim=1)
        return self.classifier(fused)

    def predict_proba(self, behavioral: torch.Tensor, tfidf: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass + softmax, for inference.

        Returns:
            Probabilities tensor of shape (batch, num_classes), rows sum to 1.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(behavioral, tfidf, embedding)
            return torch.softmax(logits, dim=1)

    def architecture_summary(self) -> dict:
        """
        Introspect the actual layers to recover hidden_dims/num_classes/dropout,
        rather than trusting values passed at construction time. Used at save
        time so checkpoints always self-describe their true architecture even
        if the caller's bookkeeping (e.g. a config dict) has drifted.

        Returns:
            Dict with keys: hidden_dims, num_classes, dropout.
        """
        linear_layers = [m for m in self.classifier if isinstance(m, nn.Linear)]
        hidden_dims = [layer.out_features for layer in linear_layers[:-1]]
        num_classes = linear_layers[-1].out_features
        dropout_layers = [m for m in self.classifier if isinstance(m, nn.Dropout)]
        dropout = dropout_layers[0].p if dropout_layers else 0.0
        return {"hidden_dims": hidden_dims, "num_classes": num_classes, "dropout": dropout}


def save_checkpoint(model: HybridFusionClassifier, filepath: str) -> None:
    """
    Save the model's weights and self-introspected architecture to disk.

    Args:
        model: Trained HybridFusionClassifier.
        filepath: Output path (e.g. 'models/artifacts/model.pt').
    """
    arch = model.architecture_summary()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "behavioral_dim": model.behavioral_dim,
            "tfidf_dim": model.tfidf_dim,
            "embedding_dim": model.embedding_dim,
            **arch,
        },
        filepath,
    )
    logger.info(f"Saved model checkpoint to {filepath}")


def load_checkpoint(filepath: str) -> HybridFusionClassifier:
    """
    Reconstruct a HybridFusionClassifier from a saved checkpoint, using the
    architecture recorded in the checkpoint itself.

    Args:
        filepath: Path to the saved checkpoint.

    Returns:
        The reconstructed model with weights loaded, in eval mode.

    Raises:
        ModelLoadError: If the checkpoint is missing or malformed.
    """
    try:
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    except FileNotFoundError as e:
        raise ModelLoadError(f"Model checkpoint not found at {filepath}") from e
    except Exception as e:
        raise ModelLoadError(f"Failed to load model checkpoint from {filepath}: {e}") from e

    required_keys = {"state_dict", "behavioral_dim", "tfidf_dim", "embedding_dim", "hidden_dims", "num_classes", "dropout"}
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise ModelLoadError(f"Checkpoint at {filepath} is missing expected keys: {missing}")

    model = HybridFusionClassifier(
        behavioral_dim=checkpoint["behavioral_dim"],
        tfidf_dim=checkpoint["tfidf_dim"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        num_classes=checkpoint["num_classes"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info(f"Loaded model checkpoint from {filepath}")
    return model
