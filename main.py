"""
main.py

Menu-driven CLI for the Prompt Injection Detection system. Ties together
preprocessing, TF-IDF transformation, model loading, and prediction into a
single interactive entry point.

Menu:
    1. Predict Prompt
    2. Exit

Workflow for option 1:
    User Input -> Preprocess -> TF-IDF Transform -> Load Model ->
    Prediction -> Display (Prediction, Confidence, Probability)

Usage:
    python main.py
"""

import logging

from predict import load_artifacts, predict_prompt, display_result
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Menu display
# --------------------------------------------------------------------------- #

def display_menu() -> None:
    """
    Print the main menu options to the console.
    """
    print("\n" + "=" * 50)
    print("Prompt Injection Detection System")
    print("=" * 50)
    print("1. Predict Prompt")
    print("2. Exit")


def get_menu_choice() -> str:
    """
    Prompt the user for a menu selection.

    Returns:
        The raw user input string, stripped of surrounding whitespace.
    """
    return input("Select an option (1-2): ").strip()


# --------------------------------------------------------------------------- #
# Prediction workflow
# --------------------------------------------------------------------------- #

def get_user_prompt() -> str:
    """
    Prompt the user to enter a text prompt to classify.

    Returns:
        The raw prompt text entered by the user.
    """
    return input("\nEnter the prompt to analyze: ").strip()


def handle_predict_prompt(model, vectorizer, label_encoder, stop_words, lemmatizer) -> None:
    """
    Run the full predict-prompt workflow for menu option 1:
    get user input -> preprocess -> TF-IDF transform -> predict -> display result.

    Args:
        model: Loaded trained classifier.
        vectorizer: Loaded fitted TfidfVectorizer.
        label_encoder: Loaded fitted LabelEncoder, or None if labels were numeric.
        stop_words: Pre-built stopword set (reused across calls for efficiency).
        lemmatizer: Pre-built WordNetLemmatizer (reused across calls for efficiency).
    """
    text = get_user_prompt()
    if not text:
        print("No prompt entered — returning to menu.")
        return

    # predict_prompt() internally handles: preprocess -> TF-IDF transform -> predict
    result = predict_prompt(text, model, vectorizer, label_encoder, stop_words, lemmatizer)
    display_result(result)


# --------------------------------------------------------------------------- #
# Application setup
# --------------------------------------------------------------------------- #

def initialize_system(
    model_path: str = "model.pkl",
    vectorizer_path: str = "vectorizer.pkl",
    encoder_path: str = "label_encoder.pkl",
):
    """
    Load the model, vectorizer, label encoder, and NLP resources needed for
    prediction. Called once at startup so repeated predictions don't reload
    artifacts from disk each time.

    Args:
        model_path: Path to the trained model.
        vectorizer_path: Path to the fitted TF-IDF vectorizer.
        encoder_path: Path to the fitted label encoder (optional).

    Returns:
        Tuple of (model, vectorizer, label_encoder, stop_words, lemmatizer).
    """
    print("Loading model and vectorizer...")
    model, vectorizer, label_encoder = load_artifacts(model_path, vectorizer_path, encoder_path)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    print("Ready.")
    return model, vectorizer, label_encoder, stop_words, lemmatizer


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def run_menu_loop(model, vectorizer, label_encoder, stop_words, lemmatizer) -> None:
    """
    Run the interactive menu loop until the user selects Exit.

    Args:
        model: Loaded trained classifier.
        vectorizer: Loaded fitted TfidfVectorizer.
        label_encoder: Loaded fitted LabelEncoder, or None if labels were numeric.
        stop_words: Pre-built stopword set.
        lemmatizer: Pre-built WordNetLemmatizer.
    """
    while True:
        display_menu()
        choice = get_menu_choice()

        if choice == "1":
            handle_predict_prompt(model, vectorizer, label_encoder, stop_words, lemmatizer)
        elif choice == "2":
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid option — please select 1 or 2.")


def main() -> None:
    """
    Application entry point: initialize the system, then run the menu loop.
    """
    model, vectorizer, label_encoder, stop_words, lemmatizer = initialize_system()
    run_menu_loop(model, vectorizer, label_encoder, stop_words, lemmatizer)


if __name__ == "__main__":
    main()
