<div align="center">

# 🇵🇰 Roman Urdu Sentiment Analysis

### A lightweight NLP pipeline for understanding Roman Urdu feedback.

</div>

## 💡 What It Is

A sentiment-analysis project built for the messy reality of Roman Urdu — inconsistent spelling, informal language, and code-switching. I built it to take noisy text from preprocessing all the way to a usable feedback-analysis app.

## 🛠️ Tech Stack

- **Python** · scikit-learn
- **TF-IDF (unigrams + bigrams)**
- **Logistic Regression**
- **Streamlit**
- pandas · NumPy

## ⚙️ How It Works

```text
Roman Urdu Text
      ↓
Cleaning + Normalization
      ↓
TF-IDF Features
      ↓
Logistic Regression
      ↓
Sentiment + Probability
```

The app supports both **single-text prediction** and **bulk CSV feedback analysis**, including negative-feedback and complaint-keyword views.

## 🚀 Run Locally

```bash
git clone https://github.com/moizaiqbal40-ops/-Roman-Urdu-Sentiment-Analysis.git
cd -Roman-Urdu-Sentiment-Analysis
pip install -r requirements.txt
python src/train.py
streamlit run streamlit_app.py
```

For a quick CLI prediction:

```bash
python src/predict.py --text "yeh cheez bohat zabardast hai"
```

## ✨ What I Learned / Challenges

The biggest challenge was handling inconsistent Roman Urdu spelling without losing useful sentiment signals, which led me to build preprocessing specifically around the language's noisy text patterns.
