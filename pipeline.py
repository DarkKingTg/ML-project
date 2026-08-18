"""
training/pipeline.py

Orchestrates the full BehaveGuard pipeline end to end:
    Load Dataset -> Clean -> Behavioral Features -> TF-IDF ->
    Embeddings -> Train Hybrid Model -> Evaluate -> Save Artifacts

This is the single entry point that ties together dataset/, preprocessing/,
features/, models/, training/, and evaluation/ into one reproducible run.
"""

from pathlib import Path

from torch.utils.data import DataLoader

from config.settings import load_config, get_output_path
from dataset.loader import load_raw_dataset, split_dataset
from preprocessing.text_cleaner import TextCleaner
from features.behavioral_features import extract_behavioral_features_batch, FEATURE_DIM
from features import tfidf_features
from features.transformer_embeddings import build_embedder
from models.fusion_model import HybridFusionClassifier, save_checkpoint
from models.hybrid_dataset import HybridFeatureDataset
from training.trainer import train_model, compute_class_weights, resolve_device
from evaluation.evaluator import (
    get_predictions, compute_full_metrics, print_metrics,
    plot_confusion_matrix, plot_roc_curve,
)
from utils.logger import get_logger, configure_logging
from utils.exceptions import BehaveGuardError

logger = get_logger(__name__)


def ensure_output_dirs(cfg: dict) -> None:
    """Create all output directories referenced in config['output'] if they don't exist."""
    for key, path in cfg["output"].items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def run_pipeline(config_path: str = "config/config.yaml") -> dict:
    """
    Run the complete BehaveGuard training pipeline end to end.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict with keys: test_metrics, embedder_backend (which backend was
        actually used -- relevant since 'auto' may have fallen back).

    Raises:
        BehaveGuardError (or a subclass): If any pipeline stage fails. The
            specific exception type identifies which stage failed.
    """
    cfg = load_config(config_path)
    configure_logging(cfg["logging"]["level"], cfg["logging"].get("log_file"))
    ensure_output_dirs(cfg)

    logger.info("=== Stage 1/6: Load Dataset ===")
    df = load_raw_dataset(cfg["data"]["raw_path"], cfg["data"]["text_column"], cfg["data"]["label_column"])
    train_df, test_df = split_dataset(df, cfg["data"]["label_column"], cfg["data"]["test_size"], cfg["data"]["random_state"])
    # Further split train into train/val for early-stopping-style best-model selection
    train_df, val_df = split_dataset(train_df, cfg["data"]["label_column"], 0.1, cfg["data"]["random_state"])

    logger.info("=== Stage 2/6: Clean Dataset ===")
    cleaner = TextCleaner(cfg)
    train_cleaned = cleaner.clean_batch(train_df[cfg["data"]["text_column"]])
    val_cleaned = cleaner.clean_batch(val_df[cfg["data"]["text_column"]])
    test_cleaned = cleaner.clean_batch(test_df[cfg["data"]["text_column"]])

    logger.info("=== Stage 3/6: Feature Engineering (Behavioral + TF-IDF + Embeddings) ===")
    # Behavioral features use RAW text (see features/behavioral_features.py docstring)
    train_behavioral = extract_behavioral_features_batch(train_df[cfg["data"]["text_column"]])
    val_behavioral = extract_behavioral_features_batch(val_df[cfg["data"]["text_column"]])
    test_behavioral = extract_behavioral_features_batch(test_df[cfg["data"]["text_column"]])

    vectorizer = tfidf_features.build_vectorizer(cfg)
    train_tfidf = tfidf_features.fit_transform(vectorizer, train_cleaned)
    val_tfidf = tfidf_features.transform(vectorizer, val_cleaned)
    test_tfidf = tfidf_features.transform(vectorizer, test_cleaned)

    embedder = build_embedder(cfg)
    logger.info(f"Embedding backend in use: {type(embedder).__name__}")
    train_embedding = embedder.embed(train_df[cfg["data"]["text_column"]].tolist(), batch_size=cfg["embeddings"]["batch_size"])
    val_embedding = embedder.embed(val_df[cfg["data"]["text_column"]].tolist(), batch_size=cfg["embeddings"]["batch_size"])
    test_embedding = embedder.embed(test_df[cfg["data"]["text_column"]].tolist(), batch_size=cfg["embeddings"]["batch_size"])

    logger.info("=== Stage 4/6: Train Hybrid Neural Network ===")
    train_labels = train_df[cfg["data"]["label_column"]].to_numpy()
    val_labels = val_df[cfg["data"]["label_column"]].to_numpy()
    test_labels = test_df[cfg["data"]["label_column"]].to_numpy()

    train_dataset = HybridFeatureDataset(train_behavioral, train_tfidf, train_embedding, train_labels)
    val_dataset = HybridFeatureDataset(val_behavioral, val_tfidf, val_embedding, val_labels)
    test_dataset = HybridFeatureDataset(test_behavioral, test_tfidf, test_embedding, test_labels)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = HybridFusionClassifier(
        behavioral_dim=FEATURE_DIM,
        tfidf_dim=train_tfidf.shape[1],
        embedding_dim=train_embedding.shape[1],
        hidden_dims=cfg["model"]["hidden_dims"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )

    class_weights = None
    if cfg["training"]["use_class_weights"]:
        class_weights = compute_class_weights(train_labels, cfg["model"]["num_classes"])

    model, best_val_metrics, _ = train_model(model, train_loader, val_loader, cfg, class_weights)
    logger.info(f"Best validation metrics: {best_val_metrics}")

    logger.info("=== Stage 5/6: Evaluation (Test Set) ===")
    device = resolve_device(cfg["training"]["device"])
    class_names = ["Safe", "Malicious"] if cfg["model"]["num_classes"] == 2 else None
    y_true, y_pred, y_proba = get_predictions(model, test_loader, device)
    test_metrics = compute_full_metrics(y_true, y_pred, y_proba, class_names=class_names)
    print_metrics(test_metrics)
    print("\nClassification Report:\n", test_metrics["classification_report"])

    out_cfg = cfg["output"]
    if class_names:
        plot_confusion_matrix(test_metrics["confusion_matrix"], class_names, out_cfg["confusion_matrix_path"])
        plot_roc_curve(y_true, y_proba, test_metrics["roc_auc"], out_cfg["roc_curve_path"])

    import json
    with open(out_cfg["metrics_path"], "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Saved test metrics to {out_cfg['metrics_path']}")

    logger.info("=== Stage 6/6: Save Artifacts ===")
    save_checkpoint(model, out_cfg["model_path"])
    tfidf_features.save_vectorizer(vectorizer, out_cfg["vectorizer_path"])
    embedder.save_tokenizer(out_cfg["embedder_dir"])

    import yaml
    with open(out_cfg["config_snapshot_path"], "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info(f"Saved config snapshot to {out_cfg['config_snapshot_path']}")

    logger.info("Pipeline completed successfully.")
    return {"test_metrics": test_metrics, "embedder_backend": type(embedder).__name__}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the full BehaveGuard training pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    try:
        run_pipeline(args.config)
    except BehaveGuardError as e:
        logger.error(f"Pipeline failed: {e}")
        raise
