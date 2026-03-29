# debug_raw_text.py
import pdfplumber
from pathlib import Path

# Pick any one PDF from your folders
PDF_PATH = r"C:\Users\cn_55\Desktop\上课文件\2 - F550\F550\Project\exercise1\transcripts\industrial\2024-Apr-18-SNA.N-140391027706-Transcript.pdf"

with pdfplumber.open(PDF_PATH) as pdf:
    # Print first 3 pages raw text
    for i, page in enumerate(pdf.pages[:3]):
        print(f"\n{'='*60}")
        print(f"PAGE {i+1}")
        print('='*60)
        text = page.extract_text()
        if text:
            print(text)