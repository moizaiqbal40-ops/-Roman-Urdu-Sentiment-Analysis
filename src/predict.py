"""
predict.py
----------
Command-line demo: type a Roman Urdu sentence and get the predicted
sentiment (Positive / Negative / Neutral) with confidence scores.

Usage:
    python src/predict.py
    python src/predict.py --text "yeh cheez bohat achi hai"
"""

import os
import argparse
import joblib

from preprocess import preprocess

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sentiment_pipeline.joblib")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run `python src/train.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_sentiment(pipeline, text: str):
    clean = preprocess(text)
    pred = pipeline.predict([clean])[0]
    proba = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_
    scores = dict(zip(classes, proba))
    return pred, scores


def main():
    parser = argparse.ArgumentParser(description="Roman Urdu Sentiment Predictor")
    parser.add_argument("--text", type=str, help="Roman Urdu sentence to analyze")
    args = parser.parse_args()

    pipeline = load_model()

    if args.text:
        pred, scores = predict_sentiment(pipeline, args.text)
        print(f"\nText     : {args.text}")
        print(f"Sentiment: {pred}")
        print("Scores   :", {k: round(v, 3) for k, v in scores.items()})
        return

    print("Roman Urdu Sentiment Analyzer — type 'exit' to quit\n")
    while True:
        text = input("Enter a sentence: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if not text:
            continue
        pred, scores = predict_sentiment(pipeline, text)
        rounded = {k: round(v, 3) for k, v in scores.items()}
        print(f"  -> Sentiment: {pred}  |  Scores: {rounded}")


if __name__ == "__main__":
    main()
