# step3_forward_looking.py
import json, re
import pandas as pd
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────
PARSED_DIR = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\parsed"
OUTPUT_DIR = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\forward_looking"

# ── Sub-sector mapping via yfinance ───────────────────────
def build_subsector_map(tickers: list) -> dict:
    """Fetch GICS sub-sector for each ticker from yfinance."""
    mapping = {}
    print("Fetching sub-sector info from yfinance...")
    for tkr in tqdm(tickers, desc="yfinance lookup"):
        try:
            info = yf.Ticker(tkr).info
            mapping[tkr] = {
                "sub_sector": info.get("industry", "Unknown"),
                "sector_yf":  info.get("sector",   "Unknown"),
            }
        except Exception:
            mapping[tkr] = {"sub_sector": "Unknown", "sector_yf": "Unknown"}
    return mapping

# ── LM wordlists ──────────────────────────────────────────
# Loughran-McDonald (2011) — condensed core lists
LM_POSITIVE = set("""
able accomplish accomplish achieve adequate advance advances
advantageous attractive best better beneficial boast capabilities
capable comfortable confidence confident creative deliver
dependable efficient effective enhance exceed excellent
exceptional expand favorable gain growth high highest
ideal improve improved improvement increasing innovative
leading optimal outstanding positive proactive productive
profitable profitability promising recover recovery reliable
reward strong stronger succeed success successful superior
sustain value win
""".split())

LM_NEGATIVE = set("""
abandon adverse against barrier challenge close costly cut
decline decrease default deficit delay difficult difficulty
disappoint dispute disrupt doubt fail failure fear
hardship harm impair inadequate insufficient late loss
lower miss obstacles reduce restructure risk shortfall
slow struggle uncertain unfavorable unlikely weak weakness
worry worse worst
""".split())

LM_UNCERTAINTY = set("""
abeyance ambiguous anticipate appear approximately assume
believe conceivable contingent could depend doubt estimate
expect fluctuate guidance if likely may might
outlook pending perhaps possible possibly predict probable
risk seem should uncertain uncertainty unpredictable
variable whether
""".split())

# ── Forward-looking triggers ──────────────────────────────
FL_RE = re.compile(
    r"\b(will|expect|anticipate|forecast|project|plan|intend|aim|target|"
    r"outlook|guidance|going forward|next quarter|next year|future|"
    r"we believe|we see|we think|look to|look ahead|continue to|"
    r"remain confident|positioned to|opportunity|growth ahead|"
    r"in the coming|over the next|by end of|pipeline|runway)\b",
    re.IGNORECASE
)

BL_RE = re.compile(
    r"\b(last quarter|last year|in Q[1-4]|as of|reported|delivered|"
    r"achieved|we saw|we had|we posted|we recorded|increased by|"
    r"decreased by|compared to|versus prior|year-over-year|"
    r"quarter-over-quarter)\b",
    re.IGNORECASE
)

# Guidance section markers
GUIDANCE_RE = re.compile(
    r"\b(guid(e|ance)|outlook|target|forecast|full.?year|next quarter|"
    r"fiscal \d{4}|FY\d{2,4}|H[12] \d{4})\b",
    re.IGNORECASE
)

SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

# ── Helpers ───────────────────────────────────────────────
def split_sentences(text: str) -> list:
    return [s.strip() for s in SENT_RE.split(text) if len(s.strip()) > 20]

def score_lm(sentence: str) -> dict:
    """Return LM positive, negative, uncertainty word counts."""
    words = re.findall(r"[a-zA-Z]+", sentence.lower())
    return {
        "lm_pos":         sum(1 for w in words if w in LM_POSITIVE),
        "lm_neg":         sum(1 for w in words if w in LM_NEGATIVE),
        "lm_uncertainty": sum(1 for w in words if w in LM_UNCERTAINTY),
    }

def is_forward_looking(sentence: str) -> bool:
    return bool(FL_RE.search(sentence)) and not bool(BL_RE.search(sentence))

def is_guidance_sentence(sentence: str) -> bool:
    return bool(GUIDANCE_RE.search(sentence))

# ── Main ──────────────────────────────────────────────────
def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(Path(PARSED_DIR).glob("*.json"))
    print(f"Found {len(json_files)} parsed transcripts")

    # Build sub-sector map from all unique tickers
    all_tickers = set()
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            all_tickers.add(json.load(f)["ticker"])

    subsector_map = build_subsector_map(sorted(all_tickers))

    # Save sub-sector map for reference
    pd.DataFrame([
        {"ticker": k, **v} for k, v in subsector_map.items()
    ]).to_csv(output_dir / "subsector_map.csv", index=False)
    print(f"Sub-sector map saved ({len(subsector_map)} tickers)")

    # ── Extract forward-looking sentences ─────────────────
    records = []

    for jf in tqdm(json_files, desc="Extracting FL sentences"):
        with open(jf, encoding="utf-8") as f:
            doc = json.load(f)

        ticker  = doc["ticker"]
        quarter = doc["quarter"]
        sector  = doc["sector"]
        date    = doc["date"]
        sub     = subsector_map.get(ticker, {})
        sub_sector = sub.get("sub_sector", "Unknown")

        # Mark removal companies for survivorship bias tracking
        is_removal = (sector == "Removal")

        for seg in doc["segments"]:
            spk_type = seg["speaker_type"]
            section  = seg["section"]

            # Only score management and analyst segments
            if spk_type not in ("management", "analyst"):
                continue

            for sent in split_sentences(seg["text"]):
                if not is_forward_looking(sent):
                    continue

                lm = score_lm(sent)

                records.append({
                    "ticker":       ticker,
                    "date":         date,
                    "quarter":      quarter,
                    "sector":       sector,
                    "sub_sector":   sub_sector,
                    "is_removal":   is_removal,
                    "speaker":      seg["speaker"],
                    "speaker_type": spk_type,
                    "section":      section,
                    "is_guidance":  is_guidance_sentence(sent),
                    "sentence":     sent,
                    "lm_pos":       lm["lm_pos"],
                    "lm_neg":       lm["lm_neg"],
                    "lm_uncertainty": lm["lm_uncertainty"],
                })

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "all_fl_sentences.csv", index=False, encoding="utf-8-sig")

    # ── Summary stats ──────────────────────────────────────
    print(f"\nTotal FL sentences : {len(df)}")
    print(f"Avg per transcript : {len(df) / len(json_files):.1f}")
    print(f"\nBy sector + speaker_type:")
    print(df.groupby(["sector", "speaker_type"])["sentence"].count().to_string())
    print(f"\nGuidance sentences : {df['is_guidance'].sum()}")
    print(f"Removal company sentences : {df['is_removal'].sum()}")
    print(f"\nTop sub-sectors (IT):")
    print(df[df["sector"]=="IT"]["sub_sector"].value_counts().head(8).to_string())
    print(f"\nTop sub-sectors (Industrials):")
    print(df[df["sector"]=="Industrials"]["sub_sector"].value_counts().head(8).to_string())

if __name__ == "__main__":
    main()