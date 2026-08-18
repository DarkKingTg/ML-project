"""
features/behavioral_features.py

Hand-crafted behavioral / statistical feature extraction for prompt
injection and jailbreak detection. Fifteen features capturing surface-level
signals -- length, casing, entropy, jailbreak vocabulary, markdown abuse,
etc. -- that complement the semantic signal from TF-IDF and transformer
embeddings.

IMPORTANT: these features operate on RAW text, not the cleaned/lemmatized
text used by the TF-IDF branch -- casing, punctuation density, and URLs are
signal here, not noise to be stripped.
"""

import re
import math
import string
from collections import Counter

import numpy as np

from utils.logger import get_logger
from utils.exceptions import FeatureExtractionError

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Lexicons (intentionally small/illustrative -- extend for production use)
# --------------------------------------------------------------------------- #

SECURITY_KEYWORDS = [
    "system", "prompt", "admin", "root", "override", "bypass", "restriction",
    "rule", "policy", "developer", "unlock", "unrestricted", "sudo", "token",
    "credential", "password", "api key", "secret",
]

ROLEPLAY_INDICATORS = [
    "pretend", "act as", "you are now", "roleplay", "imagine you are",
    "from now on you", "your new persona", "in this fictional",
    "let's play a game", "you will play the role",
]

INSTRUCTION_OVERRIDE_PHRASES = [
    "ignore previous instructions", "ignore all previous", "disregard the above",
    "forget everything", "new instructions", "system override",
    "ignore your instructions", "do not follow", "override your programming",
]

JAILBREAK_KEYWORDS = [
    "dan", "do anything now", "jailbreak", "no restrictions", "unfiltered",
    "uncensored", "without limitations", "no ethical", "no moral", "no safety",
    "bypass safety", "disable safety", "developer mode",
]

FEATURE_NAMES = [
    "prompt_length", "token_count", "uppercase_ratio", "digit_ratio",
    "punctuation_ratio", "special_char_count", "keyword_frequency",
    "roleplay_indicator_count", "instruction_override_count",
    "jailbreak_keyword_count", "entropy", "repetition_score",
    "url_count", "code_block_count", "markdown_count",
]

FEATURE_DIM = len(FEATURE_NAMES)


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of the character distribution, in bits."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _repetition_score(text: str) -> float:
    """Fraction of tokens that repeat an earlier token (flags padding/flooding attacks)."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(tokens)


def _markdown_count(text: str) -> int:
    """Count markdown syntax elements: headers, bold/italic, links, inline code."""
    headers = len(re.findall(r"^#{1,6}\s", text, flags=re.MULTILINE))
    bold_italic = len(re.findall(r"\*{1,2}[^*]+\*{1,2}", text))
    links = len(re.findall(r"\[[^\]]+\]\([^)]+\)", text))
    inline_code = len(re.findall(r"`[^`]+`", text))
    return headers + bold_italic + links + inline_code


def _code_block_count(text: str) -> int:
    """Count markdown code fences (```), pairing fences into blocks."""
    fences = text.count("```")
    return fences // 2 if fences >= 2 else fences


def extract_behavioral_features(text: str) -> np.ndarray:
    """
    Extract all 15 behavioral features from a single raw prompt string.

    Args:
        text: Raw prompt text (NOT preprocessed).

    Returns:
        1D numpy array of shape (FEATURE_DIM,), dtype float32, ordered per FEATURE_NAMES.

    Raises:
        FeatureExtractionError: If extraction fails unexpectedly on this input.
    """
    try:
        text = str(text)
        text_lower = text.lower()
        letters = [c for c in text if c.isalpha()]

        features = [
            len(text),                                                        # prompt_length
            len(text.split()),                                                # token_count
            (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0,  # uppercase_ratio
            (sum(1 for c in text if c.isdigit()) / len(text)) if text else 0.0,           # digit_ratio
            (sum(1 for c in text if c in string.punctuation) / len(text)) if text else 0.0,  # punctuation_ratio
            sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in set(string.punctuation)),  # special_char_count
            sum(text_lower.count(kw) for kw in SECURITY_KEYWORDS),               # keyword_frequency
            sum(text_lower.count(p) for p in ROLEPLAY_INDICATORS),               # roleplay_indicator_count
            sum(text_lower.count(p) for p in INSTRUCTION_OVERRIDE_PHRASES),      # instruction_override_count
            sum(text_lower.count(kw) for kw in JAILBREAK_KEYWORDS),              # jailbreak_keyword_count
            _shannon_entropy(text),                                             # entropy
            _repetition_score(text),                                            # repetition_score
            len(re.findall(r"(https?://\S+|www\.\S+)", text)),                   # url_count
            _code_block_count(text),                                            # code_block_count
            _markdown_count(text),                                              # markdown_count
        ]
        return np.array(features, dtype=np.float32)
    except Exception as e:
        raise FeatureExtractionError(
            f"Behavioral feature extraction failed on input {str(text)[:80]!r}: {e}"
        ) from e


def extract_behavioral_features_batch(texts) -> np.ndarray:
    """
    Extract behavioral features for a batch of raw prompts.

    Args:
        texts: Iterable of raw prompt strings.

    Returns:
        2D numpy array of shape (n_samples, FEATURE_DIM), dtype float32.

    Raises:
        FeatureExtractionError: If extraction fails on the batch.
    """
    texts = list(texts)
    if not texts:
        raise FeatureExtractionError("Cannot extract behavioral features from an empty batch")
    logger.info(f"Extracting behavioral features for {len(texts)} prompts")
    return np.vstack([extract_behavioral_features(t) for t in texts])
