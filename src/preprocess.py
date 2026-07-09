"""
preprocess.py
--------------
Text cleaning utilities for Roman Urdu text.

Roman Urdu has no fixed spelling standard (e.g. "acha", "acha", "achaa" all
mean the same thing), so cleaning focuses on:
  - lowercasing
  - stripping URLs, mentions, hashtags, punctuation, digits
  - removing extra whitespace
  - removing a small hand-curated Roman Urdu stopword list
  - collapsing repeated characters (e.g. "zabardastttt" -> "zabardastt")
"""

import re
import string

# A compact stopword list for Roman Urdu (function words that carry
# little sentiment signal). This list can be extended over time.
ROMAN_URDU_STOPWORDS = {
    "hai", "hain", "ho", "hun", "hy", "ka", "ki", "ke", "ko", "ky",
    "ku", "kr", "kar", "ny", "ne", "se", "sy", "ye", "yeh", "wo", "woh",
    "aur", "or", "ap", "aap", "tum", "tu", "mein", "main", "mai", "hum",
    "hm", "un", "in", "is", "us", "es", "wo", "jo", "jis", "jab", "tab",
    "agar", "magar", "lekin", "per", "par", "to", "toh", "bhi", "b",
    "nahi", "nai", "nhi", "na", "kya", "kia", "q", "kyun", "kyu",
    "liye", "lye", "se", "wala", "wali", "wale", "raha", "rahi", "rhy",
    "gaya", "gayi", "gy", "tha", "thi", "thy", "hoga", "hogi", "hongy",
    "iss", "uss", "yahan", "wahan", "abhi", "phir", "sab", "koi", "kuch",
}


def clean_text(text: str) -> str:
    """Lowercase + strip noise from a single Roman Urdu sentence."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                        # mentions
    text = re.sub(r"#\w+", " ", text)                        # hashtags
    text = re.sub(r"[0-9]+", " ", text)                      # digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)                # zabardasttttt -> zabardastt
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    tokens = [t for t in text.split() if t not in ROMAN_URDU_STOPWORDS]
    return " ".join(tokens)


def preprocess(text: str, drop_stopwords: bool = True) -> str:
    text = clean_text(text)
    if drop_stopwords:
        text = remove_stopwords(text)
    return text


if __name__ == "__main__":
    sample = "Yeh product bilkul zabardastttt hai!!! www.example.com @user #great"
    print("raw   :", sample)
    print("clean :", preprocess(sample))
