"""
app.py
------
Small Flask web demo for the Roman Urdu Sentiment Analyzer.

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import joblib
from flask import Flask, render_template_string, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import preprocess  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sentiment_pipeline.joblib")

app = Flask(__name__)
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Roman Urdu Sentiment Analyzer</title>
  <style>
    body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0;
           display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
    .card { background: #1e293b; padding: 40px; border-radius: 16px; width: 480px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
    h1 { font-size: 22px; margin-bottom: 4px; }
    p.sub { color: #94a3b8; margin-top: 0; font-size: 14px; }
    textarea { width: 100%; padding: 12px; border-radius: 8px; border: none;
               background: #0f172a; color: #e2e8f0; font-size: 15px; box-sizing: border-box; }
    button { margin-top: 12px; padding: 10px 22px; border: none; border-radius: 8px;
             background: #6366f1; color: white; font-size: 15px; cursor: pointer; }
    button:hover { background: #4f46e5; }
    .result { margin-top: 20px; padding: 16px; border-radius: 10px; }
    .Positive { background: #14532d; }
    .Negative { background: #7f1d1d; }
    .Neutral  { background: #334155; }
    .scores { font-size: 13px; color: #cbd5e1; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🇵🇰 Roman Urdu Sentiment Analyzer</h1>
    <p class="sub">TF-IDF + Logistic Regression trained on 20k+ tagged Roman Urdu sentences</p>
    <form method="POST">
      <textarea name="text" rows="4" placeholder="e.g. yeh cheez bohat zabardast hai">{{ text or '' }}</textarea><br>
      <button type="submit">Analyze</button>
    </form>
    {% if prediction %}
    <div class="result {{ prediction }}">
      <strong>Sentiment: {{ prediction }}</strong>
      <div class="scores">{{ scores }}</div>
    </div>
    {% endif %}
    {% if not model_loaded %}
    <p style="color:#f87171; margin-top:16px;">⚠️ Model not found — run <code>python src/train.py</code> first.</p>
    {% endif %}
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    scores = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "")
        if model and text.strip():
            clean = preprocess(text)
            prediction = model.predict([clean])[0]
            proba = model.predict_proba([clean])[0]
            scores = ", ".join(f"{c}: {p:.2f}" for c, p in zip(model.classes_, proba))
    return render_template_string(
        PAGE, prediction=prediction, scores=scores, text=text, model_loaded=model is not None
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
