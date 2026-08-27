# 🇵🇰 Roman Urdu Sentiment Analysis

A sentiment classifier for **Roman Urdu** (Urdu written in Latin script — the way
most Pakistanis actually text and comment online). Most sentiment analysis
portfolios only use English datasets; this project targets a genuinely
low-resource, under-served language setting.

## Why this project

- Roman Urdu has **no fixed spelling standard** ("acha", "achaa", "acha" all mean
  the same thing) — this alone breaks most off-the-shelf English NLP tooling.
- Trained on a **real, manually-tagged dataset of 20,000+ sentences** collected
  from e-commerce reviews, Facebook comments, and tweets (Sharf, Z. — *Roman Urdu
  Dataset*, hosted on the UCI Machine Learning Repository / GitHub).
- Demonstrates the full ML pipeline: data cleaning → feature engineering →
  model training → evaluation → deployment as a web demo.

## Results

| Metric | Score |
|---|---|
| Accuracy | ~64% (3-class: Positive / Negative / Neutral) |
| Weighted F1 | ~0.64 |

See `models/confusion_matrix.png` and `models/metrics.json` after training for
the full breakdown. (Baseline of random-guess on 3 imbalanced classes is ~33-44%,
so this is a solid signal — plenty of room to push further with word embeddings
or a transformer like XLM-R / mBERT, noted in Future Work below.)

## Project structure

```
roman-urdu-sentiment-analysis/
├── data/
│   └── roman_urdu_dataset.csv     # 20k+ labeled Roman Urdu sentences
├── src/
│   ├── preprocess.py               # text cleaning + Roman Urdu stopwords
│   ├── train.py                    # trains TF-IDF + Logistic Regression
│   └── predict.py                  # CLI: type a sentence, get sentiment
├── models/                         # saved model + metrics (generated)
├── app.py                          # Flask web demo
├── requirements.txt
└── README.md
```

## How it works

1. **Preprocessing** (`src/preprocess.py`) — lowercases text, strips URLs/
   mentions/hashtags/digits/punctuation, collapses repeated letters
   ("zabardasttttt" → "zabardastt"), and removes a hand-curated list of Roman
   Urdu stopwords (hai, hain, ka, ki, aur, etc.).
2. **Feature extraction** — TF-IDF over unigrams + bigrams (up to 20,000
   features).
3. **Model** — Logistic Regression with balanced class weights (handles the
   Neutral-heavy class imbalance in the dataset).
4. **Evaluation** — accuracy, weighted F1, full classification report, and a
   confusion matrix plot.

## Setup

```bash
git clone <your-repo-url>
cd roman-urdu-sentiment-analysis
pip install -r requirements.txt
```

## Train the model

```bash
python src/train.py
```

This creates:
- `models/sentiment_pipeline.joblib` — the trained pipeline
- `models/metrics.json` — accuracy / F1 / full classification report
- `models/confusion_matrix.png` — visual breakdown

## Try it out

**Command line:**
```bash
python src/predict.py --text "yeh cheez bohat zabardast hai"
# Sentiment: Positive

python src/predict.py --text "bakwas hai yeh to"
# Sentiment: Negative
```

Or run it interactively:
```bash
python src/predict.py
```

**Web demo:**
```bash
python app.py
```
Then open `http://127.0.0.1:5000` and type any Roman Urdu sentence.

## Dataset credit

Dataset compiled by Zareen Sharf, hosted at
[Smat26/Roman-Urdu-Dataset](https://github.com/Smat26/Roman-Urdu-Dataset)
(GPL-3.0), originally on the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set).
Full credit to the original author — this project builds a classifier and
demo layer on top of it.

## Future work

- Swap TF-IDF for word embeddings (fastText has pretrained Urdu vectors) or
  fine-tune a multilingual transformer (mBERT / XLM-R) for higher accuracy.
- Expand the stopword list and add stemming/lemmatization rules specific to
  Roman Urdu.
- Add a browser extension that analyzes sentiment on Roman Urdu social media
  comments in real time.
