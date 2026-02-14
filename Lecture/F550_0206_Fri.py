import re
import json
import requests
from bs4 import BeautifulSoup
from readability import Document

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

URL = "https://www.bbc.co.uk/news/articles/c0jvjp14qjzo"

def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text

def extract_main_text(html: str) -> str:
    doc = Document(html)
    main_html = doc.summary()
    soup = BeautifulSoup(main_html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text

def clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

def label_from_compound(compound: float, pos_th=0.05, neg_th=-0.05) -> str:
    if compound >= pos_th:
        return "positive"
    if compound <= neg_th:
        return "negative"
    return "neutral"

def main():
    # 1) down VADER lexicon and sentence split model
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download('punkt_tab')

    sia = SentimentIntensityAnalyzer()

    # 2) reference dictionary
    vader_path = nltk.data.find("sentiment/vader_lexicon.zip")
    print(f"[Dictionary] Using NLTK VADER lexicon at: {vader_path}")

    sample_words = ["good", "bad", "happy", "sad", "terrible", "excellent"]
    print("[Dictionary] Sample entries (word -> valence):")
    for w in sample_words:
        print(f"  {w}: {sia.lexicon.get(w)}")

    # 3) fetch web and extract main body
    html = fetch_html(URL)
    text = extract_main_text(html)
    print(f"\n[Info] Extracted main text length: {len(text)} characters")

    # 4) split sentences
    raw_sents = nltk.sent_tokenize(text)
    sents = [clean_sentence(s) for s in raw_sents if clean_sentence(s)]

    # 5) label sentences
    counts = {"negative": 0, "neutral": 0, "positive": 0}
    per_sentence = []

    for i, s in enumerate(sents, start=1):
        scores = sia.polarity_scores(s)  # {'neg','neu','pos','compound'}
        label = label_from_compound(scores["compound"])
        counts[label] += 1
        per_sentence.append({
            "idx": i,
            "label": label,
            "compound": scores["compound"],
            "neg": scores["neg"],
            "neu": scores["neu"],
            "pos": scores["pos"],
            "sentence": s
        })

    total = sum(counts.values()) or 1
    ratio = {k: v / total for k, v in counts.items()}
    overall = max(ratio.items(), key=lambda x: x[1])[0]

    # 6) outcomes
    print("\n=== Sentence-level counts ===")
    print(json.dumps(counts, indent=2))
    print("\n=== Sentence-level ratios ===")
    print(json.dumps(ratio, indent=2))
    print(f"\nOverall article label (by sentence majority): {overall}")


if __name__ == "__main__":
    main()

