"""
streamlit_app.py
-----------------
Roman Urdu Customer Feedback Intelligence — a business-facing tool, not
just a toy classifier.

Two modes:
  1. Quick Check   - analyze a single sentence instantly.
  2. Bulk Analysis - upload a CSV of customer reviews/comments (Roman Urdu)
     and get an automatic feedback report: sentiment breakdown, the most
     urgent negative comments to act on, and the most common complaint
     keywords — the kind of thing a small business or startup could
     actually use to triage customer feedback instead of reading every
     comment by hand.
"""

import os
import sys
from collections import Counter

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import preprocess  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sentiment_pipeline.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "roman_urdu_dataset.csv")

st.set_page_config(
    page_title="Roman Urdu Feedback Intelligence",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ #
# Styling
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f6faf7 0%, #ffffff 35%); }

    .hero {
        background: linear-gradient(120deg, #01411C 0%, #046A38 55%, #0e8a4b 100%);
        padding: 34px 40px;
        border-radius: 18px;
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 10px 30px rgba(1,65,28,0.25);
    }
    .hero h1 { margin: 0 0 6px 0; font-size: 30px; }
    .hero p { margin: 0; opacity: 0.9; font-size: 15px; }

    .metric-card {
        background: white;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #eef2ef;
        text-align: center;
    }
    .metric-card .value { font-size: 26px; font-weight: 700; }
    .metric-card .label { font-size: 13px; color: #6b7280; margin-top: 2px; }

    .sentiment-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 15px;
    }
    .badge-Positive { background: #d7f4e0; color: #0a7a37; }
    .badge-Negative { background: #fbdada; color: #b91c1c; }
    .badge-Neutral  { background: #e5e7eb; color: #374151; }

    .complaint-card {
        background: #fff7f7;
        border-left: 4px solid #dc2626;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 14px;
    }
    div[data-testid="stFileUploader"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🇵🇰 Roman Urdu Feedback Intelligence</h1>
        <p>Turn raw Roman Urdu customer comments into an instant sentiment report —
        built for businesses that get feedback in Roman Urdu but have no easy way to triage it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model (first run trains it, ~30s)...")
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

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


def classify(texts):
    clean = [preprocess(t) for t in texts]
    preds = model.predict(clean)
    probs = model.predict_proba(clean)
    confidences = probs.max(axis=1)
    return preds, confidences


def top_keywords(texts, n=12):
    """Most frequent non-stopword tokens across a list of raw texts."""
    counter = Counter()
    for t in texts:
        cleaned = preprocess(t, drop_stopwords=True)
        for word in cleaned.split():
            if len(word) > 2:
                counter[word] += 1
    return counter.most_common(n)


tab1, tab2 = st.tabs(["⚡ Quick Check", "📊 Bulk Feedback Analysis"])

# ------------------------------------------------------------------ #
# Tab 1 — single sentence
# ------------------------------------------------------------------ #
with tab1:
    col_a, col_b = st.columns([3, 2])
    with col_a:
        text = st.text_area(
            "Type a Roman Urdu sentence:",
            placeholder="e.g. delivery bohat late hui, product theek tha",
            height=110,
        )
        check_clicked = st.button("Analyze sentiment", type="primary")

        if check_clicked and text.strip():
            pred, conf = classify([text])
            pred, conf = pred[0], conf[0]
            st.markdown(
                f'<span class="sentiment-badge badge-{pred}">{pred} · {conf:.0%} confident</span>',
                unsafe_allow_html=True,
            )
        elif check_clicked:
            st.warning("Pehle koi sentence likho.")

    with col_b:
        st.markdown("**Try an example:**")
        for ex in [
            "delivery bohat late hui, product theek tha",
            "customer service ne bilkul waste kar diya time",
            "yeh cheez zabardast hai, dobara order karunga",
        ]:
            if st.button(ex, key=ex):
                pred, conf = classify([ex])
                st.markdown(
                    f'**"{ex}"** → <span class="sentiment-badge badge-{pred[0]}">{pred[0]}</span>',
                    unsafe_allow_html=True,
                )

# ------------------------------------------------------------------ #
# Tab 2 — bulk analysis (the "real problem solved" part)
# ------------------------------------------------------------------ #
with tab2:
    st.markdown(
        "Upload a CSV of customer comments/reviews (Roman Urdu) and get an "
        "instant feedback report — sentiment breakdown, the most urgent "
        "negative comments to act on first, and the most common complaint "
        "keywords."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("...or use a sample dataset to see how it works")

    df_input = None
    if uploaded is not None:
        df_input = pd.read_csv(uploaded)
    elif use_sample:
        sample_df = pd.read_csv(DATA_PATH)
        df_input = sample_df.sample(min(300, len(sample_df)), random_state=1)[["text"]]

    if df_input is not None:
        text_col = st.selectbox("Which column has the comment text?", df_input.columns.tolist())
        texts = df_input[text_col].dropna().astype(str).tolist()

        if st.button("Run analysis", type="primary"):
            with st.spinner(f"Analyzing {len(texts)} comments..."):
                preds, confs = classify(texts)
                result_df = pd.DataFrame({
                    "comment": texts,
                    "sentiment": preds,
                    "confidence": confs,
                })

            counts = result_df["sentiment"].value_counts()
            total = len(result_df)
            pos_pct = counts.get("Positive", 0) / total * 100
            neg_pct = counts.get("Negative", 0) / total * 100
            neu_pct = counts.get("Neutral", 0) / total * 100

            st.markdown("#### Overview")
            m1, m2, m3, m4 = st.columns(4)
            for col, label, value, color in [
                (m1, "Total comments", total, "#111"),
                (m2, "Positive", f"{pos_pct:.0f}%", "#0a7a37"),
                (m3, "Negative", f"{neg_pct:.0f}%", "#b91c1c"),
                (m4, "Neutral", f"{neu_pct:.0f}%", "#374151"),
            ]:
                col.markdown(
                    f'<div class="metric-card"><div class="value" style="color:{color}">{value}</div>'
                    f'<div class="label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            chart_col, keyword_col = st.columns(2)

            with chart_col:
                st.markdown("##### Sentiment distribution")
                fig, ax = plt.subplots(figsize=(4, 4))
                colors = {"Positive": "#0a7a37", "Negative": "#dc2626", "Neutral": "#9ca3af"}
                order = [s for s in ["Positive", "Neutral", "Negative"] if s in counts.index]
                ax.pie(
                    [counts[s] for s in order],
                    labels=order,
                    autopct="%1.0f%%",
                    colors=[colors[s] for s in order],
                    startangle=90,
                    wedgeprops={"edgecolor": "white", "linewidth": 2},
                )
                ax.axis("equal")
                st.pyplot(fig)

            with keyword_col:
                st.markdown("##### Most common complaint keywords")
                negative_texts = result_df[result_df["sentiment"] == "Negative"]["comment"].tolist()
                if negative_texts:
                    kw = top_keywords(negative_texts, n=10)
                    kw_df = pd.DataFrame(kw, columns=["keyword", "count"]).set_index("keyword")
                    st.bar_chart(kw_df, color="#dc2626")
                else:
                    st.info("No negative comments found — nothing to flag 🎉")

            st.markdown("#### 🚩 Most urgent comments to review")
            st.caption("Highest-confidence negative comments — likely the strongest complaints.")
            urgent = (
                result_df[result_df["sentiment"] == "Negative"]
                .sort_values("confidence", ascending=False)
                .head(8)
            )
            if urgent.empty:
                st.success("No urgent negative feedback in this batch.")
            else:
                for _, row in urgent.iterrows():
                    st.markdown(
                        f'<div class="complaint-card">🔴 <b>{row["confidence"]:.0%}</b> negative — {row["comment"]}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Full results")
            st.dataframe(result_df, use_container_width=True, height=300)
            st.download_button(
                "⬇️ Download annotated results (CSV)",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="feedback_analysis_results.csv",
                mime="text/csv",
            )

with st.expander("ℹ️ About this project"):
    st.markdown(
        """
        Most sentiment tools only cover English, but a huge amount of real customer
        feedback in Pakistan — Facebook comments, WhatsApp, e-commerce reviews — is
        written in **Roman Urdu**, which has no fixed spelling standard. This tool
        cleans and classifies that text, and packages it into the kind of report a
        small business or support team could actually use: what % of feedback is
        negative, which comments need attention first, and what people are
        complaining about most.

        **Model**: TF-IDF + Logistic Regression, trained on 20,000+ manually tagged
        Roman Urdu sentences.
        Dataset credit: Zareen Sharf — Roman Urdu Dataset
        ([Smat26/Roman-Urdu-Dataset](https://github.com/Smat26/Roman-Urdu-Dataset)).
        """
    )
