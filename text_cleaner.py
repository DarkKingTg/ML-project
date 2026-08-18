"""
preprocessing/text_cleaner.py

Text cleaning and normalization for the BehaveGuard pipeline: lowercasing,
URL/HTML/punctuation/number stripping, tokenization, stopword removal, and
lemmatization. Produces the `cleaned_text` used by the TF-IDF branch.

NOTE: behavioral features (features/behavioral_features.py) intentionally
operate on RAW, uncleaned text -- signals like uppercase ratio and
punctuation density are destroyed by this cleaning step, so that module
never calls into this one.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from utils.logger import get_logger
from utils.exceptions import PreprocessingError

logger = get_logger(__name__)

_NLTK_RESOURCES = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
    "corpora/omw-1.4": "omw-1.4",
}


def ensure_nltk_resources() -> None:
    """
    Download required NLTK resources if not already present locally. Safe to
    call repeatedly. Raises PreprocessingError if downloads fail (e.g. no
    network access and resources aren't cached).
    """
    for path, name in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                logger.info(f"Downloading NLTK resource: {name}")
                nltk.download(name, quiet=True)
            except Exception as e:
                raise PreprocessingError(
                    f"Failed to download required NLTK resource '{name}': {e}"
                ) from e


class TextCleaner:
    """
    Encapsulates the text cleaning pipeline as a stateful object (holds the
    stopword set and lemmatizer so they're built once, not per call).
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: Full configuration dict (reads cfg['preprocessing']).
        """
        ensure_nltk_resources()
        self.cfg = cfg["preprocessing"]
        self.stop_words = set(stopwords.words("english")) if self.cfg.get("remove_stopwords", True) else set()
        self.lemmatizer = WordNetLemmatizer() if self.cfg.get("lemmatize", True) else None

    def clean(self, text: str) -> str:
        """
        Apply the configured cleaning pipeline to a single text string.

        Args:
            text: Raw input text.

        Returns:
            Cleaned, tokenized-and-rejoined text string.

        Raises:
            PreprocessingError: If cleaning fails unexpectedly on this input.
        """
        try:
            text = str(text)

            if self.cfg.get("lowercase", True):
                text = text.lower()
            if self.cfg.get("remove_urls", True):
                text = re.sub(r"(https?://\S+|www\.\S+)", " ", text)
            if self.cfg.get("remove_html", True):
                text = re.sub(r"<.*?>", " ", text)
            if self.cfg.get("remove_punctuation", True):
                text = text.translate(str.maketrans("", "", string.punctuation))
            if self.cfg.get("remove_numbers", True):
                text = re.sub(r"\d+", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            tokens = word_tokenize(text)
            if self.stop_words:
                tokens = [t for t in tokens if t not in self.stop_words]
            if self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

            return " ".join(tokens)
        except Exception as e:
            raise PreprocessingError(f"Failed to clean text (input preview: {str(text)[:80]!r}): {e}") from e

    def clean_batch(self, texts) -> list:
        """
        Apply cleaning to a batch of texts.

        Args:
            texts: Iterable of raw text strings.

        Returns:
            List of cleaned text strings, same order as input.
        """
        return [self.clean(t) for t in texts]
