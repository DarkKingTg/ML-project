"""
predict.py

Run inference with a trained prompt-injection detection model:
loads the saved TF-IDF vectorizer and Logistic Regression model, applies the
exact same preprocessing used at training time, and classifies a prompt as
Safe or Malicious with confidence and full probability scores.

Usage:
    python predict.py
    (then type a prompt at the interactive input prompt)

    or as a module:
    from predict import load_artifacts, predict_prompt
    model, vectorizer, label_encoder = load_artifacts()
    result = predict_prompt("Ignore all previous instructions...", model, vectorizer, label_encoder)
    print(result)
"""

import logging

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from preprocessing import preprocess_pipeline, download_nltk_resources

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Loading artifacts
# --------------------------------------------------------------------------- #

def load_model(filepath: str = "model.pkl"):
    """
    Load a trained classifier from disk.

    Args:
        filepath: Path to the pickled model (default 'model.pkl').

    Returns:
        The fitted classifier (e.g. LogisticRegression).
    """
    logger.info(f"Loading model from {filepath}")
    return joblib.load(filepath)


def load_vectorizer(filepath: str = "vectorizer.pkl"):
    """
    Load a fitted TF-IDF vectorizer from disk.

    Args:
        filepath: Path to the pickled vectorizer (default 'vectorizer.pkl').

    Returns:
        The fitted TfidfVectorizer.
    """
    logger.info(f"Loading vectorizer from {filepath}")
    return joblib.load(filepath)


def load_label_encoder(filepath: str = "label_encoder.pkl"):
    """
    Load a fitted LabelEncoder from disk, if one was saved during training.

    Args:
        filepath: Path to the pickled label encoder (default 'label_encoder.pkl').

    Returns:
        The fitted LabelEncoder, or None if no encoder file is found (this is
        expected when training labels were already numeric, e.g. 0/1).
    """
    try:
        encoder = joblib.load(filepath)
        logger.info(f"Loaded label encoder from {filepath}")
        return encoder
    except FileNotFoundError:
        logger.info(
            f"No label encoder found at {filepath} — assuming numeric labels "
            f"(0 = Safe, 1 = Malicious)"
        )
        return None


def load_artifacts(
    model_path: str = "model.pkl",
    vectorizer_path: str = "vectorizer.pkl",
    encoder_path: str = "label_encoder.pkl",
):
    """
    Load all artifacts needed for inference: model, vectorizer, and (optionally)
    label encoder. Also ensures required NLTK resources are available.

    Args:
        model_path: Path to the trained model.
        vectorizer_path: Path to the fitted TF-IDF vectorizer.
        encoder_path: Path to the fitted label encoder (optional).

    Returns:
        Tuple of (model, vectorizer, label_encoder). label_encoder is None
        if labels were numeric at training time.
    """
    download_nltk_resources()
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)
    label_encoder = load_label_encoder(encoder_path)
    return model, vectorizer, label_encoder


# --------------------------------------------------------------------------- #
# Preprocessing (must exactly match training-time preprocessing)
# --------------------------------------------------------------------------- #

def preprocess_input(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """
    Apply the same preprocessing pipeline used during training
    (clean -> tokenize -> remove stopwords -> lemmatize) to a raw user prompt.

    Args:
        text: Raw user input string.
        stop_words: Set of stopwords to remove.
        lemmatizer: An initialized WordNetLemmatizer instance.

    Returns:
        Preprocessed text, ready to be passed to the TF-IDF vectorizer.
    """
    return preprocess_pipeline(text, stop_words, lemmatizer)


# --------------------------------------------------------------------------- #
# Label interpretation
# --------------------------------------------------------------------------- #

def label_to_name(label, label_encoder) -> str:
    """
    Convert a raw predicted label into a human-readable 'Safe' / 'Malicious' string.

    Args:
        label: Raw predicted label (numeric 0/1, or an encoded class from
            label_encoder.inverse_transform).
        label_encoder: Fitted LabelEncoder used at training time, or None if
            labels were already numeric (0 = Safe, 1 = Malicious).

    Returns:
        'Safe' or 'Malicious'.
    """
    if label_encoder is not None:
        raw_label = label_encoder.inverse_transform([label])[0]
    else:
        raw_label = label

    raw_label_str = str(raw_label).strip().lower()
    if raw_label_str in ("1", "malicious", "true", "unsafe", "injection"):
        return "Malicious"
    return "Safe"


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def predict_prompt(
    text: str,
    model,
    vectorizer,
    label_encoder=None,
    stop_words: set = None,
    lemmatizer: WordNetLemmatizer = None,
) -> dict:
    """
    Classify a single raw prompt as Safe or Malicious.

    Args:
        text: Raw user input string.
        model: Trained classifier (e.g. LogisticRegression) with predict_proba.
        vectorizer: Fitted TfidfVectorizer used at training time.
        label_encoder: Fitted LabelEncoder used at training time, or None if
            labels were numeric.
        stop_words: Optional pre-built stopword set (built automatically if None).
        lemmatizer: Optional pre-built WordNetLemmatizer (built automatically if None).

    Returns:
        Dict with keys:
            'prompt': original raw text
            'cleaned_text': preprocessed text passed to the vectorizer
            'prediction': 'Safe' or 'Malicious'
            'confidence': probability of the predicted class (float, 0-1)
            'probabilities': dict mapping each class name to its probability
    """
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    cleaned = preprocess_input(text, stop_words, lemmatizer)
    X = vectorizer.transform([cleaned])

    predicted_class = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    prediction = label_to_name(predicted_class, label_encoder)
    confidence = float(max(proba))

    # Build a human-readable probability breakdown per class
    probabilities = {}
    for cls, p in zip(model.classes_, proba):
        class_name = label_to_name(cls, label_encoder)
        probabilities[class_name] = float(p)

    return {
        "prompt": text,
        "cleaned_text": cleaned,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def display_result(result: dict) -> None:
    """
    Pretty-print a prediction result to the console.

    Args:
        result: Dict returned by predict_prompt().
    """
    print(f"\nPrompt: {result['prompt']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Probability scores:")
    for class_name, prob in sorted(result["probabilities"].items()):
        print(f"  {class_name}: {prob:.4f}")


# --------------------------------------------------------------------------- #
# Interactive entry point
# --------------------------------------------------------------------------- #

def run_interactive(
    model_path: str = "model.pkl",
    vectorizer_path: str = "vectorizer.pkl",
    encoder_path: str = "label_encoder.pkl",
) -> None:
    """
    Load artifacts and run an interactive loop that accepts user prompts from
    stdin and prints predictions until the user types 'exit' or 'quit'.

    Args:
        model_path: Path to the trained model.
        vectorizer_path: Path to the fitted TF-IDF vectorizer.
        encoder_path: Path to the fitted label encoder (optional).
    """
    model, vectorizer, label_encoder = load_artifacts(model_path, vectorizer_path, encoder_path)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    print("Prompt Injection Detector — type a prompt to classify it ('exit' to quit)")
    while True:
        text = input("\nEnter prompt: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if not text:
            continue
        result = predict_prompt(text, model, vectorizer, label_encoder, stop_words, lemmatizer)
        display_result(result)


if __name__ == "__main__":
    run_interactive()
