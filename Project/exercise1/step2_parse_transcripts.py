# step2_parse_transcripts.py
import os, re, json
import pdfplumber
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\transcripts"
OUTPUT_DIR = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\parsed"

FOLDER_SECTOR = {
    "industrial":  "Industrials",
    "software&IT": "IT",
    "tech":        "IT",
    "removal":     "Removal",
}

MONTH_MAP = {
    "Jan":"01","Feb":"02","Mar":"03","Apr":"04",
    "May":"05","Jun":"06","Jul":"07","Aug":"08",
    "Sep":"09","Oct":"10","Nov":"11","Dec":"12"
}

ANALYST_KEYWORDS = {
    "analyst","research","capital","securities","partners","bank",
    "asset","investment","equity","morgan","goldman","barclays",
    "jpmorgan","ubs","citi","wells fargo","bernstein","oppenheimer",
    "jefferies","cowen","baird","needham","rosenblatt","lightshed",
    "division","oppenheimer","guggenheim","loop capital","td cowen",
    "new street","baird","raymond james"
}

# ── Helpers ───────────────────────────────────────────────
def parse_filename(filename):
    parts = Path(filename).stem.split("-")
    try:
        ticker = parts[3].split(".")[0].split("_")[0]
        month  = MONTH_MAP.get(parts[1], "00")
        date   = f"{parts[0]}-{month}-{parts[2].zfill(2)}"
    except Exception:
        ticker, date = "UNKNOWN", "UNKNOWN"
    return ticker, date

def date_to_quarter(date):
    try:
        y, m = date[:4], int(date[5:7])
        if   m <= 3:  return f"{int(y)-1}Q4"
        elif m <= 6:  return f"{y}Q1"
        elif m <= 9:  return f"{y}Q2"
        else:         return f"{y}Q3"
    except Exception:
        return "UNKNOWN"

def classify_speaker(role, company):
    combined = (role + " " + company).lower()
    if any(k in combined for k in ANALYST_KEYWORDS):
        return "analyst"
    if any(k in combined for k in ["ceo","cfo","coo","president","vp","vice president",
                                    "director","officer","head","treasurer","controller"]):
        return "management"
    return "other"

# ── Core: Refinitiv speaker line detector ─────────────────
# Real format from PDF:
#   "Sara M. Verbsky Snap-on Incorporated - VP of IR"
#   "Nicholas T. Pinchuk Snap-on Incorporated - Chairman, CEO & President"
#   "Bret David Jordan Jefferies LLC, Research Division - MD & Equity Analyst"
#
# Pattern: <Name (title-case words)> <Company> - <Role>
# Name ends when we hit the company (all-caps acronym OR title-case org words before the dash)
# Simplest reliable approach: split on " - " and work backwards from the dash

SPEAKER_LINE_RE = re.compile(
    # Name: one or more title-case words (may include initials like "T.")
    r"^([A-Z][a-zA-Z\.\-]+(?:\s+[A-Z][a-zA-Z\.\-]+){1,4})"
    # Company: everything up to " - "
    r"\s+(.+?)"
    # Role: after the dash
    r"\s+-\s+(.+)$"
)

# Lines to skip (headers, footers, page markers)
SKIP_RE = re.compile(
    r"refinitiv|streetevents|©\d{4}|contact us|"
    r"prohibited without|republication|all rights reserved|"
    r"^\d+$|^page \d+",
    re.IGNORECASE
)

def parse_transcript(raw_text):
    lines   = raw_text.split("\n")
    segments = []
    cur_lines = []
    speaker = company = role = spk_type = None
    section = "presentation"

    # Build participant set from CORPORATE PARTICIPANTS block
    # (helps us later confirm speaker lines vs random title-case sentences)
    participants = set()
    in_participants = False
    for line in lines:
        line_s = line.strip()
        if "CORPORATE PARTICIPANTS" in line_s.upper():
            in_participants = True
            continue
        if in_participants:
            if any(x in line_s.upper() for x in ["CONFERENCE CALL", "PRESENTATION",
                                                   "QUESTIONS AND ANSWERS", "DISCLAIMER"]):
                in_participants = False
                continue
            m = SPEAKER_LINE_RE.match(line_s)
            if m:
                participants.add(m.group(1).strip())

    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        if SKIP_RE.search(line_s):
            continue

        # Section markers
        if "QUESTIONS AND ANSWERS" in line_s.upper():
            section = "qa"
            continue
        if "PRESENTATION" in line_s.upper() and len(line_s) < 30:
            section = "presentation"
            continue
        if any(x in line_s.upper() for x in ["CORPORATE PARTICIPANTS",
                                               "CONFERENCE CALL PARTICIPANTS",
                                               "DISCLAIMER"]):
            continue

        # Speaker line detection
        # Strategy: match the regex AND require name is in participants list
        # OR the line appears after a blank line and matches the pattern
        m = SPEAKER_LINE_RE.match(line_s)
        if m:
            name_candidate = m.group(1).strip()
            # Accept if name seen in participants block, or if role looks right
            role_candidate    = m.group(3).strip()
            company_candidate = m.group(2).strip()
            looks_like_speaker = (
                name_candidate in participants or
                any(k in role_candidate.lower() for k in
                    ["ceo","cfo","analyst","president","vp","director",
                     "officer","head","ir","treasurer","operator"])
            )
            if looks_like_speaker:
                # Save previous segment
                if speaker and cur_lines:
                    segments.append({
                        "speaker":      speaker,
                        "company":      company,
                        "role":         role,
                        "speaker_type": spk_type,
                        "section":      section,
                        "text":         " ".join(cur_lines).strip()
                    })
                    cur_lines = []
                speaker  = name_candidate
                company  = company_candidate
                role     = role_candidate
                spk_type = classify_speaker(role, company)
                continue

        if speaker:
            cur_lines.append(line_s)

    # Last segment
    if speaker and cur_lines:
        segments.append({
            "speaker":      speaker,
            "company":      company,
            "role":         role,
            "speaker_type": spk_type,
            "section":      section,
            "text":         " ".join(cur_lines).strip()
        })

    return segments

# ── Main ──────────────────────────────────────────────────
def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pdfs = []
    for folder, sector in FOLDER_SECTOR.items():
        p = Path(BASE_DIR) / folder
        if p.exists():
            for pdf in p.glob("*.pdf"):
                all_pdfs.append((pdf, sector))

    print(f"Found {len(all_pdfs)} PDFs")

    index, failed = [], []

    for pdf_path, sector in tqdm(all_pdfs, desc="Parsing"):
        ticker, date = parse_filename(pdf_path.name)
        quarter      = date_to_quarter(date)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                raw = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            failed.append(pdf_path.name)
            continue

        if not raw.strip():
            failed.append(pdf_path.name)
            continue

        segments = parse_transcript(raw)

        result = {
            "ticker": ticker, "date": date, "quarter": quarter,
            "sector": sector, "filename": pdf_path.name,
            "segments": segments
        }

        with open(output_dir / (pdf_path.stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        index.append({
            "ticker":       ticker, "date": date,
            "quarter":      quarter, "sector": sector,
            "n_segments":   len(segments),
            "n_management": sum(1 for s in segments if s["speaker_type"] == "management"),
            "n_analyst":    sum(1 for s in segments if s["speaker_type"] == "analyst"),
        })

    idx = pd.DataFrame(index).sort_values(["sector","ticker","date"])
    idx.to_csv(output_dir / "index.csv", index=False)

    print(f"\nDone: {len(index)} parsed, {len(failed)} failed")
    print(f"Companies : {idx['ticker'].nunique()}")
    print(f"\nSegment stats (should be > 0):")
    print(f"  Avg segments per transcript : {idx['n_segments'].mean():.1f}")
    print(f"  Transcripts with 0 segments : {(idx['n_segments']==0).sum()}")
    print(f"\nBy sector:")
    print(idx.groupby("sector")[["n_management","n_analyst"]].mean().round(1).to_string())

    print(f"\n✅ All transcripts converted -> {OUTPUT_DIR}")

if __name__ == "__main__":
    main()