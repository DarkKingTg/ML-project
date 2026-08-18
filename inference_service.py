"""
api/inference_service.py

Loads trained artifacts (model, TF-IDF vectorizer, embedder) once and exposes
a single `predict()` method used by both the CLI and the HTTP API layer.
Keeping this separate from the HTTP framework means the prediction logic is
testable and usable without spinning up a web server.
"""

from pathlib import Path

import numpy as np
import torch

from config.settings import load_config
from preprocessing.text_cleaner import TextCleaner
from features.behavioral_features import extract_behavioral_features
from features import tfidf_features
from features.transformer_embeddings import build_embedder
from models.fusion_model import load_checkpoint
from utils.logger import get_logger
from utils.exceptions import PredictionError, ModelLoadError

logger = get_logger(__name__)

CLASS_NAMES = {0: "Safe", 1: "Malicious"}


class InferenceService:
    """
    Loads all artifacts needed for prediction exactly once, then serves
    predict() calls cheaply. Intended to be instantiated a single time at
    process startup (e.g. as a module-level singleton in the API layer).
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Load config and all trained artifacts.

        Args:
            config_path: Path to config.yaml (must match the config used at
                training time, or dimensions won't line up).

        Raises:
            ModelLoadError: If any artifact fails to load.
        """
        self.cfg = load_config(config_path)
        out_cfg = self.cfg["output"]

        logger.info("Loading InferenceService artifacts...")
        self.cleaner = TextCleaner(self.cfg)
        self.vectorizer = tfidf_features.load_vectorizer(out_cfg["vectorizer_path"])
        self.model = load_checkpoint(out_cfg["model_path"])
        self.device = "cpu"  # inference served on CPU by default; adjust if deploying with GPU
        self.model.to(self.device)

        self.embedder = self._load_embedder(out_cfg["embedder_dir"])
        logger.info(f"InferenceService ready (embedder backend: {type(self.embedder).__name__})")

    def _load_embedder(self, embedder_dir: str):
        """
        Load the embedding backend used at training time. If the saved
        artifacts indicate the hashing fallback was used (marker file
        present), reconstruct a HashingEmbedder with matching dimensions
        rather than attempting (and failing) to load a real transformer.

        Args:
            embedder_dir: Directory where the embedder/tokenizer was saved.

        Returns:
            A TransformerEmbedder or HashingEmbedder instance.

        Raises:
            ModelLoadError: If the embedder cannot be reconstructed.
        """
        from features.transformer_embeddings import HashingEmbedder, TransformerEmbedder

        marker = Path(embedder_dir) / "HASHING_BACKEND_USED.txt"
        if marker.exists():
            logger.info("Detected hashing-backend marker; reconstructing HashingEmbedder for inference")
            return HashingEmbedder(hidden_size=self.cfg["embeddings"]["hidden_size"])

        try:
            import torch as _torch
            from transformers import AutoTokenizer, AutoModel

            embedder = TransformerEmbedder.__new__(TransformerEmbedder)
            embedder.model_name = self.cfg["embeddings"]["model_name"]
            embedder.max_length = self.cfg["embeddings"]["max_length"]
            embedder.device = "cpu"
            embedder._torch = _torch
            embedder.tokenizer = AutoTokenizer.from_pretrained(embedder_dir)
            embedder.model = AutoModel.from_pretrained(self.cfg["embeddings"]["model_name"])
            embedder.model.to(embedder.device)
            embedder.model.eval()
            embedder.hidden_size = embedder.model.config.hidden_size
            return embedder
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load transformer embedder from {embedder_dir}: {e}"
            ) from e

    def predict(self, text: str) -> dict:
        """
        Classify a single raw prompt.

        Args:
            text: Raw prompt text.

        Returns:
            Dict with keys:
                'prompt': the original input text
                'prediction': 'Safe' or 'Malicious'
                'confidence': probability of the predicted class
                'probabilities': dict mapping class name -> probability

        Raises:
            PredictionError: If any stage of inference fails.
        """
        if not isinstance(text, str) or not text.strip():
            raise PredictionError("Prediction input must be a non-empty string")

        try:
            cleaned = self.cleaner.clean(text)
            behavioral = extract_behavioral_features(text).reshape(1, -1)
            tfidf = tfidf_features.transform(self.vectorizer, [cleaned])
            embedding = self.embedder.embed([text])

            behavioral_t = torch.tensor(behavioral, dtype=torch.float32).to(self.device)
            tfidf_t = torch.tensor(tfidf, dtype=torch.float32).to(self.device)
            embedding_t = torch.tensor(embedding, dtype=torch.float32).to(self.device)

            proba = self.model.predict_proba(behavioral_t, tfidf_t, embedding_t)[0].cpu().numpy()
            predicted_idx = int(np.argmax(proba))

            return {
                "prompt": text,
                "prediction": CLASS_NAMES.get(predicted_idx, str(predicted_idx)),
                "confidence": float(proba[predicted_idx]),
                "probabilities": {CLASS_NAMES.get(i, str(i)): float(p) for i, p in enumerate(proba)},
            }
        except PredictionError:
            raise
        except Exception as e:
            raise PredictionError(f"Prediction failed for input {text[:80]!r}: {e}") from e

    def predict_batch(self, texts: list) -> list:
        """
        Classify a batch of prompts. Simply loops predict() -- acceptable at
        the batch sizes typical of a security-review API; swap for true
        batched tensor ops if throughput becomes a bottleneck.

        Args:
            texts: List of raw prompt strings.

        Returns:
            List of prediction dicts, one per input, same order.
        """
        return [self.predict(t) for t in texts]
