# step2_parse_transcripts.py
import os, re, json
import pdfplumber
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\transcripts"
OUTPUT_DIR = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\parsed"

# Map each subfolder to a sector label
FOLDER_SECTOR = {
    "industrial":  "Industrials",
    "software&IT": "IT",
    "tech":        "IT",
    "removal":     "Removal",   # survivorship bias sample
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
    "jefferies","cowen","baird","needham","rosenblatt","lightShed"
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
    if any(k in company.lower() for k in ANALYST_KEYWORDS):
        return "analyst"
    if any(k in role.lower() for k in ["ceo","cfo","coo","president","vp","director","officer","head"]):
        return "management"
    return "other"

SPEAKER_RE = re.compile(r"^([A-Z][a-zA-Z\s\.\-]+?)\s{2,}(.+?)\s*[-–]\s*(.+)$")

def parse_transcript(raw_text):
    segments, cur_lines = [], []
    speaker = company = role = spk_type = None
    section = "presentation"

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line: continue
        if "refinitiv" in line.lower() or "©" in line or "streetevents" in line.lower(): continue
        if "QUESTIONS AND ANSWERS" in line.upper(): section = "qa";           continue
        if "PRESENTATION"          in line.upper() and len(line) < 30: section = "presentation"; continue

        m = SPEAKER_RE.match(line)
        if m:
            if speaker and cur_lines:
                segments.append({"speaker": speaker, "company": company,
                                  "role": role, "speaker_type": spk_type,
                                  "section": section,
                                  "text": " ".join(cur_lines).strip()})
                cur_lines = []
            speaker  = m.group(1).strip()
            company  = m.group(2).strip()
            role     = m.group(3).strip()
            spk_type = classify_speaker(role, company)
        elif speaker:
            cur_lines.append(line)

    if speaker and cur_lines:
        segments.append({"speaker": speaker, "company": company,
                          "role": role, "speaker_type": spk_type,
                          "section": section,
                          "text": " ".join(cur_lines).strip()})
    return segments

# ── Main ──────────────────────────────────────────────────

def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pdfs = []
    for folder, sector in FOLDER_SECTOR.items():
        folder_path = Path(BASE_DIR) / folder
        if not folder_path.exists():
            print(f"Folder not found: {folder_path}")
            continue
        for pdf in folder_path.glob("*.pdf"):
            all_pdfs.append((pdf, sector))

    print(f"Found {len(all_pdfs)} PDFs across {len(FOLDER_SECTOR)} folders")

    index, failed = [], []

    for pdf_path, sector in tqdm(all_pdfs, desc="Parsing"):
        ticker, date = parse_filename(pdf_path.name)
        quarter      = date_to_quarter(date)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                raw = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception as e:
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

        out_file = output_dir / (pdf_path.stem + ".json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        index.append({
            "ticker":   ticker, "date": date, "quarter": quarter,
            "sector":   sector, "filename": pdf_path.name,
            "n_segments":  len(segments),
            "n_management": sum(1 for s in segments if s["speaker_type"] == "management"),
            "n_analyst":    sum(1 for s in segments if s["speaker_type"] == "analyst"),
        })

    # Save index
    idx = pd.DataFrame(index).sort_values(["sector","ticker","date"])
    idx.to_csv(output_dir / "index.csv", index=False)

    print(f"\nDone: {len(index)} parsed, {len(failed)} failed")
    print(f"Companies : {idx['ticker'].nunique()}")
    print(f"Sectors   :\n{idx['sector'].value_counts()}")
    print(f"Quarters  :\n{idx['quarter'].value_counts().sort_index()}")
    if failed:
        print(f"\nFailed files: {failed}")

if __name__ == "__main__":
    main()