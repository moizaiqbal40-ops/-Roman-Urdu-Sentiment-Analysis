"""
train.py
--------
Trains a TF-IDF + Logistic Regression sentiment classifier on the
Roman Urdu dataset and saves the fitted pipeline + label encoder + a
confusion-matrix plot + a metrics report.

Usage:
    python src/train.py
"""

import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from preprocess import preprocess

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "roman_urdu_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "sentiment"])
    df["clean_text"] = df["text"].apply(preprocess)
    df = df[df["clean_text"].str.len() > 0]
    return df


def main():
    print("Loading dataset...")
    df = load_data()
    print(f"Rows after cleaning: {len(df)}")
    print(df["sentiment"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment"],
        test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=5.0,
        )),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nAccuracy       : {acc:.4f}")
    print(f"Weighted F1    : {f1:.4f}\n")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save metrics
    metrics = {
        "accuracy": acc,
        "weighted_f1": f1,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    labels = sorted(df["sentiment"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Roman Urdu Sentiment Classifier")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print(f"Confusion matrix saved to models/confusion_matrix.png")

    # Save model
    model_path = os.path.join(MODEL_DIR, "sentiment_pipeline.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
