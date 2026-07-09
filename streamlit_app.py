"""
streamlit_app.py
-----------------
Streamlit web demo for the Roman Urdu Sentiment Analyzer.
Deployable directly on Streamlit Community Cloud.

If the trained model isn't found (e.g. fresh clone without models/ committed),
this app trains it automatically on first load so it always works out of the box.
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import preprocess  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sentiment_pipeline.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "roman_urdu_dataset.csv")

st.set_page_config(page_title="Roman Urdu Sentiment Analyzer", page_icon="🇵🇰", layout="centered")


@st.cache_resource(show_spinner="Loading model (training on first run may take ~30s)...")
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # Fallback: train on the fly if the model file isn't present in the repo
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "sentiment"])
    df["clean_text"] = df["text"].apply(preprocess)
    df = df[df["clean_text"].str.len() > 0]

    X_train, _, y_train, _ = train_test_split(
        df["clean_text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=5.0)),
    ])
    pipeline.fit(X_train, y_train)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


model = get_model()

st.title("🇵🇰 Roman Urdu Sentiment Analyzer")
st.caption(
    "TF-IDF + Logistic Regression trained on 20,000+ tagged Roman Urdu sentences "
    "(Facebook comments, e-commerce reviews, tweets)."
)

text = st.text_area(
    "Type a Roman Urdu sentence:",
    placeholder="e.g. yeh cheez bohat zabardast hai",
    height=100,
)

col1, col2 = st.columns([1, 3])
analyze_clicked = col1.button("Analyze", type="primary")

if analyze_clicked and text.strip():
    clean = preprocess(text)
    pred = model.predict([clean])[0]
    proba = model.predict_proba([clean])[0]
    scores = dict(zip(model.classes_, proba))

    color = {"Positive": "green", "Negative": "red", "Neutral": "gray"}.get(pred, "blue")
    st.markdown(f"### Sentiment: :{color}[{pred}]")

    df_scores = pd.DataFrame({"Sentiment": list(scores.keys()), "Confidence": list(scores.values())})
    st.bar_chart(df_scores.set_index("Sentiment"))

elif analyze_clicked:
    st.warning("Please type a sentence first.")

st.markdown("---")
st.markdown("#### Try these examples:")
examples = [
    "yeh cheez bohat zabardast hai",
    "bakwas hai yeh to bilkul",
    "theek thak hai koi khaas baat nahi",
]
ex_cols = st.columns(len(examples))
for c, ex in zip(ex_cols, examples):
    if c.button(ex):
        clean = preprocess(ex)
        pred = model.predict([clean])[0]
        st.info(f"**\"{ex}\"** → {pred}")

with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        Roman Urdu (Urdu written in Latin script) has no fixed spelling standard,
        which makes it a genuinely low-resource NLP problem — most sentiment tools
        only cover English. This project cleans and normalizes Roman Urdu text,
        then trains a TF-IDF + Logistic Regression classifier on a real, manually
        tagged dataset of 20,000+ sentences.

        Dataset credit: Zareen Sharf — Roman Urdu Dataset (UCI ML Repository /
        [Smat26/Roman-Urdu-Dataset](https://github.com/Smat26/Roman-Urdu-Dataset)).
        """
    )
