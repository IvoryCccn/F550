# debug_check.py
import json
from pathlib import Path

PARSED_DIR = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\parsed"

# Pick one file and inspect
jf = list(Path(PARSED_DIR).glob("*.json"))[0]
with open(jf, encoding="utf-8") as f:
    doc = json.load(f)

print("Keys:", list(doc.keys()))
print("Ticker:", doc.get("ticker"))
print("Sector:", doc.get("sector"))
print("Segment count:", len(doc.get("segments", [])))
print()

# Show first 3 segments
for seg in doc["segments"][:3]:
    print("--- segment ---")
    print("speaker_type:", seg.get("speaker_type"))
    print("section:", seg.get("section"))
    print("text preview:", seg.get("text","")[:200])
    print()