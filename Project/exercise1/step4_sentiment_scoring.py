# step4_sentiment_scoring.py
import json, re, time
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
import openai

# ── Config ────────────────────────────────────────────────
FL_CSV      = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\forward_looking\all_fl_sentences.csv"
OUTPUT_DIR  = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\scored"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = "gpt-4o-mini"
BATCH_SIZE_GPT = 20
MAX_RETRIES    = 3

# ── Load data ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FL_CSV, encoding="utf-8-sig")
    print(f"Loaded {len(df)} FL sentences")
    print(f"Columns: {list(df.columns)}")
    return df

# ════════════════════════════════════════════════════════
# METHOD 1: LM Lexicon (already scored in Step 3)
# Compute net sentiment score from existing lm_pos / lm_neg cols
# ════════════════════════════════════════════════════════
def score_lm(df):
    """
    LM net score = (pos - neg) / (pos + neg + 1)
    Normalised to [-1, +1]. Uncertainty score kept separate.
    """
    df = df.copy()
    df["lm_score"] = (df["lm_pos"] - df["lm_neg"]) / (
        df["lm_pos"] + df["lm_neg"] + 1
    )
    # Uncertainty score: normalised word count
    word_counts = df["sentence"].str.split().str.len().clip(lower=1)
    df["lm_uncertainty_score"] = df["lm_uncertainty"] / word_counts
    print("LM scoring done.")
    return df

# ════════════════════════════════════════════════════════
# METHOD 2: FinBERT (local, free)
# ════════════════════════════════════════════════════════
def score_finbert(df):
    """
    Run FinBERT on every sentence.
    Returns positive probability as the sentiment score (0 to 1).
    """
    print("Loading FinBERT model...")
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=-1,          # CPU; change to 0 if you have GPU
        truncation=True,
        max_length=512,
        top_k=None,         # return all three label scores
    )

    sentences = df["sentence"].tolist()
    results   = []

    print(f"Scoring {len(sentences)} sentences with FinBERT...")
    batch_size = 64
    for i in tqdm(range(0, len(sentences), batch_size), desc="FinBERT"):
        batch = sentences[i : i + batch_size]
        try:
            preds = finbert(batch)
            for pred in preds:
                scores = {p["label"].lower(): p["score"] for p in pred}
                results.append({
                    "finbert_pos":     scores.get("positive", 0.0),
                    "finbert_neg":     scores.get("negative", 0.0),
                    "finbert_neu":     scores.get("neutral",  0.0),
                    # Net score: pos - neg, range [-1, +1]
                    "finbert_score":   scores.get("positive", 0.0)
                                     - scores.get("negative", 0.0),
                })
        except Exception as e:
            print(f"  FinBERT batch error @ {i}: {e}")
            for _ in batch:
                results.append({
                    "finbert_pos": np.nan, "finbert_neg": np.nan,
                    "finbert_neu": np.nan, "finbert_score": np.nan,
                })

    finbert_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), finbert_df], axis=1)
    print("FinBERT scoring done.")
    return df

# ════════════════════════════════════════════════════════
# METHOD 3: GPT-4o-mini — intensity score 0-10
# ════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a financial analyst scoring forward-looking sentences from earnings call transcripts.

For each sentence, return a JSON array where each element has:
- "score": integer 0-10 (0 = extremely bearish/negative outlook, 5 = neutral, 10 = extremely bullish/positive outlook)
- "intensity": integer 1-3 (1 = mild tone, 2 = moderate, 3 = strong/emphatic tone)

Focus ONLY on the forward-looking sentiment about the company's future prospects.
Return ONLY the JSON array, no explanation, no markdown."""

def build_user_prompt(sentences: list) -> str:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    return f"Score these {len(sentences)} sentences:\n\n{numbered}"

def call_gpt_batch(client, sentences: list) -> list:
    """Call GPT for a batch, return list of {score, intensity} dicts."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(sentences)},
                ],
                temperature=0,
                max_tokens=len(sentences) * 15,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)
            if len(parsed) == len(sentences):
                return parsed
            # Length mismatch — pad with nulls
            while len(parsed) < len(sentences):
                parsed.append({"score": None, "intensity": None})
            return parsed[:len(sentences)]
        except Exception as e:
            print(f"  GPT attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)

    return [{"score": None, "intensity": None}] * len(sentences)

def score_gpt(df):
    """Score all sentences with GPT-4o-mini in batches."""
    client    = openai.OpenAI(api_key=OPENAI_API_KEY)
    sentences = df["sentence"].tolist()
    all_results = []

    print(f"Scoring {len(sentences)} sentences with GPT-4o-mini "
          f"(batch size={BATCH_SIZE_GPT})...")

    for i in tqdm(range(0, len(sentences), BATCH_SIZE_GPT), desc="GPT"):
        batch   = sentences[i : i + BATCH_SIZE_GPT]
        results = call_gpt_batch(client, batch)
        all_results.extend(results)
        # Polite rate-limit pause
        time.sleep(0.1)

    gpt_df = pd.DataFrame(all_results)
    gpt_df.columns = ["gpt_score", "gpt_intensity"]
    df = pd.concat([df.reset_index(drop=True), gpt_df], axis=1)
    print("GPT scoring done.")
    return df

# ════════════════════════════════════════════════════════
# Combine & save
# ════════════════════════════════════════════════════════
def save_checkpoint(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Checkpoint saved → {path}")

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()

    # ── Method 1: LM (instant, already have the counts) ──
    df = score_lm(df)
    save_checkpoint(df, output_dir / "scored_lm.csv")

    # ── Method 2: FinBERT ─────────────────────────────────
    df = score_finbert(df)
    save_checkpoint(df, output_dir / "scored_lm_finbert.csv")

    # ── Method 3: GPT-4o-mini ────────────────────────────
    df = score_gpt(df)
    save_checkpoint(df, output_dir / "scored_all.csv")

    # ── Summary stats ─────────────────────────────────────
    print("\n── Score distributions ──────────────────────────")
    for col in ["lm_score", "finbert_score", "gpt_score"]:
        print(f"\n{col}:")
        print(df[col].describe().round(3).to_string())

    print("\n── Correlation between methods ──────────────────")
    corr = df[["lm_score","finbert_score","gpt_score"]].corr().round(3)
    print(corr.to_string())

    print(f"\n✅ All scoring complete → {output_dir / 'scored_all.csv'}")

if __name__ == "__main__":
    main()