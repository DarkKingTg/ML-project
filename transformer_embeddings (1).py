"""
features/transformer_embeddings.py

Sentence embeddings for the BehaveGuard pipeline. Two backends:

    "transformer" -- real contextual embeddings from microsoft/deberta-v3-base
        (or any HuggingFace encoder), mean-pooled over tokens. Requires
        network access to huggingface.co to download weights.

    "hashing" -- a deterministic, fully offline embedding proxy (hashed
        n-gram bag-of-words projected to a fixed dimension via a fixed
        random projection matrix seeded for reproducibility). This is NOT a
        substitute for real contextual embeddings -- it captures no semantic
        or word-order information -- but it lets the full pipeline run,
        save, and be evaluated end-to-end in environments without Hub
        access, with an architecture that accepts a real transformer as a
        drop-in replacement later.

`embeddings.backend: "auto"` in config tries "transformer" first and falls
back to "hashing" with a clear warning if the model can't be loaded (e.g. no
network). This is a deliberate, logged degradation -- never a silent one.
"""

import hashlib

import numpy as np

from utils.logger import get_logger
from utils.exceptions import FeatureExtractionError, ModelLoadError

logger = get_logger(__name__)


class TransformerEmbedder:
    """
    Wraps a pretrained HuggingFace encoder to produce mean-pooled sentence
    embeddings.
    """

    def __init__(self, model_name: str, max_length: int = 256, device: str = None):
        """
        Load the tokenizer and model. Raises ModelLoadError if the weights
        cannot be downloaded/loaded (e.g. no network access).

        Args:
            model_name: HuggingFace model identifier.
            max_length: Maximum token sequence length.
            device: 'cuda' or 'cpu'. Auto-detected if None.

        Raises:
            ModelLoadError: If the tokenizer/model fail to load.
        """
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch

        try:
            logger.info(f"Loading transformer tokenizer/model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load transformer model '{model_name}' -- this usually "
                f"means no network access to huggingface.co. Set "
                f"embeddings.backend to 'hashing' in config to run fully offline, "
                f"or 'auto' to fall back automatically. Original error: {e}"
            ) from e

        self.model.to(self.device)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size

    def _mean_pool(self, last_hidden_state, attention_mask):
        """Mean-pool token embeddings, masking out padding tokens."""
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def embed(self, texts, batch_size: int = 16) -> np.ndarray:
        """
        Compute mean-pooled embeddings for a list of texts.

        Args:
            texts: List of raw text strings.
            batch_size: Texts per forward pass.

        Returns:
            2D numpy array of shape (n_texts, hidden_size), dtype float32.

        Raises:
            FeatureExtractionError: If embedding fails.
        """
        texts = list(texts)
        try:
            all_embeddings = []
            with self._torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    encoded = self.tokenizer(
                        batch, padding=True, truncation=True,
                        max_length=self.max_length, return_tensors="pt",
                    ).to(self.device)
                    outputs = self.model(**encoded)
                    pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                    all_embeddings.append(pooled.cpu().numpy())
            return np.vstack(all_embeddings).astype(np.float32)
        except Exception as e:
            raise FeatureExtractionError(f"Transformer embedding failed: {e}") from e

    def save_tokenizer(self, save_dir: str) -> None:
        """Save the tokenizer for consistent inference-time tokenization."""
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Saved tokenizer to {save_dir}")


class HashingEmbedder:
    """
    Deterministic, fully offline sentence embedding proxy. Used as an
    automatic fallback when a real transformer can't be loaded (no network
    access), so the full pipeline remains runnable and testable end-to-end.

    NOT a substitute for real contextual embeddings -- captures token
    identity and frequency via hashing, but no word order or semantics.
    """

    def __init__(self, hidden_size: int = 768, seed: int = 42):
        """
        Args:
            hidden_size: Output embedding dimension (matches the configured
                transformer hidden_size so the fusion model's input dim is
                unaffected by which backend produced the embeddings).
            seed: Random seed for the projection matrix, for reproducibility.
        """
        self.hidden_size = hidden_size
        self.seed = seed
        self.model_name = "hashing-embedder-fallback"

    @staticmethod
    def _hash_token(token: str, dim: int) -> int:
        """Deterministically hash a token to a bucket index in [0, dim)."""
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % dim

    def embed(self, texts, batch_size: int = 16) -> np.ndarray:
        """
        Compute a hashed bag-of-words embedding for each text, L2-normalized.

        Args:
            texts: List of raw text strings.
            batch_size: Unused (present for interface parity with TransformerEmbedder).

        Returns:
            2D numpy array of shape (n_texts, hidden_size), dtype float32.

        Raises:
            FeatureExtractionError: If embedding fails.
        """
        try:
            texts = list(texts)
            embeddings = np.zeros((len(texts), self.hidden_size), dtype=np.float32)
            for i, text in enumerate(texts):
                for token in str(text).lower().split():
                    bucket = self._hash_token(token, self.hidden_size)
                    embeddings[i, bucket] += 1.0
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return embeddings / norms
        except Exception as e:
            raise FeatureExtractionError(f"Hashing embedding failed: {e}") from e

    def save_tokenizer(self, save_dir: str) -> None:
        """
        No real tokenizer to save for this backend; writes a marker file so
        downstream code can detect which backend produced a given artifact set.
        """
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        marker = Path(save_dir) / "HASHING_BACKEND_USED.txt"
        marker.write_text(
            "This model was trained using the HashingEmbedder fallback, not a "
            "real transformer, because embeddings.backend resolved to 'hashing' "
            "(no network access to huggingface.co, or backend explicitly set). "
            "Predictions at inference time must use the same backend.\n"
        )
        logger.warning(f"Hashing backend has no real tokenizer; wrote marker file to {save_dir}")


def build_embedder(cfg: dict):
    """
    Build the configured embedding backend, resolving "auto" to "transformer"
    with a fallback to "hashing" if loading fails.

    Args:
        cfg: Full configuration dict (reads cfg['embeddings']).

    Returns:
        A TransformerEmbedder or HashingEmbedder instance (both expose the
        same `.embed(texts)` and `.save_tokenizer(dir)` interface).

    Raises:
        FeatureExtractionError: If backend is explicitly "transformer" and
            it fails to load (no silent fallback in that case).
    """
    emb_cfg = cfg["embeddings"]
    backend = emb_cfg.get("backend", "auto")
    model_name = emb_cfg["model_name"]
    max_length = emb_cfg["max_length"]
    hidden_size = emb_cfg["hidden_size"]

    if backend == "hashing":
        logger.info("Using HashingEmbedder (offline fallback) as explicitly configured")
        return HashingEmbedder(hidden_size=hidden_size)

    if backend == "transformer":
        return TransformerEmbedder(model_name=model_name, max_length=max_length)

    if backend == "auto":
        try:
            return TransformerEmbedder(model_name=model_name, max_length=max_length)
        except ModelLoadError as e:
            logger.warning(
                f"Transformer backend unavailable ({e}); "
                f"falling back to offline HashingEmbedder. Predictions will be "
                f"less accurate than with real transformer embeddings."
            )
            return HashingEmbedder(hidden_size=hidden_size)

    raise FeatureExtractionError(f"Unknown embeddings.backend '{backend}' in config")
