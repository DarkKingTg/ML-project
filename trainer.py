"""
training/trainer.py

Training and evaluation loop for the HybridFusionClassifier.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.logger import get_logger
from utils.exceptions import TrainingError

logger = get_logger(__name__)


def resolve_device(device_setting: str) -> str:
    """
    Resolve 'auto'/'cpu'/'cuda' to an actual device string.

    Args:
        device_setting: Value from config['training']['device'].

    Returns:
        'cuda' or 'cpu'.
    """
    if device_setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted CrossEntropyLoss.

    Args:
        labels: Array of integer class labels (training split only).
        num_classes: Total number of classes.

    Returns:
        1D tensor of shape (num_classes,), normalized to sum to num_classes.
    """
    classes, counts = np.unique(labels, return_counts=True)
    weights = np.ones(num_classes, dtype=np.float32)
    raw = 1.0 / counts
    normalized = raw * (len(classes) / raw.sum())
    for cls, w in zip(classes, normalized):
        weights[int(cls)] = w
    logger.info(f"Class weights computed: {weights.tolist()} (counts: {dict(zip(classes.tolist(), counts.tolist()))})")
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device: str) -> float:
    """
    Run one training epoch.

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    for behavioral, tfidf, embedding, labels in loader:
        behavioral, tfidf, embedding, labels = behavioral.to(device), tfidf.to(device), embedding.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(behavioral, tfidf, embedding)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loader(model, loader: DataLoader, criterion, device: str) -> dict:
    """
    Evaluate the model on a DataLoader.

    Returns:
        Dict with keys: loss, accuracy, precision, recall, f1.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for behavioral, tfidf, embedding, labels in loader:
        behavioral, tfidf, embedding, labels = behavioral.to(device), tfidf.to(device), embedding.to(device), labels.to(device)
        logits = model(behavioral, tfidf, embedding)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }


def train_model(model, train_loader, val_loader, cfg: dict, class_weights: torch.Tensor = None):
    """
    Run the full multi-epoch training loop with per-epoch validation and
    best-model tracking (by validation F1).

    Args:
        model: HybridFusionClassifier (moved to the resolved device internally).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: Full configuration dict (reads cfg['training']).
        class_weights: Optional tensor of per-class loss weights.

    Returns:
        Tuple of (model, best_val_metrics, best_state_dict):
            model: the model with weights from the BEST epoch (by val F1) restored.
            best_val_metrics: metrics dict from that best epoch.
            best_state_dict: the raw state_dict of the best epoch (for saving).

    Raises:
        TrainingError: If training fails (e.g. NaN loss) or the loaders are empty.
    """
    train_cfg = cfg["training"]
    device = resolve_device(train_cfg["device"])
    logger.info(f"Training on device: {device}")

    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise TrainingError("Cannot train with an empty train or validation dataset")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    weights = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_f1 = -1.0
    best_metrics = None
    best_state = None

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if np.isnan(train_loss):
            raise TrainingError(f"Training loss became NaN at epoch {epoch} — check learning rate / data scaling")

        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        logger.info(
            f"Epoch {epoch}/{train_cfg['epochs']} — train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"New best model at epoch {epoch} (val_f1={best_f1:.4f})")

    if best_state is None:
        raise TrainingError("Training completed but no best model was recorded")

    model.load_state_dict(best_state)
    model.eval()
    logger.info(f"Restored best model weights (val_f1={best_f1:.4f})")
    return model, best_metrics, best_state
