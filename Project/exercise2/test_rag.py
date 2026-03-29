#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:20:22 2026

@author: fabriziocoiai
"""
import pandas as pd
from llm_backend import GeminiBackend
from filing_rag import FilingRAG

llm = GeminiBackend(
    chat_model="gemini-2.0-flash",
    embed_model="gemini-embedding-001",
)

rag = FilingRAG()
rag.add_document(
    llm,
    doc_id="aapl_test_doc",
    ticker="AAPL",
    filed=pd.Timestamp("2025-05-02"),
    source="10-Q",
    text="Management discussed services growth, gross margin resilience, and ongoing capital returns."
)

chunks = rag.retrieve(
    llm,
    ticker="AAPL",
    query="Apple valuation, margins, growth, risks, capital returns",
    asof=pd.Timestamp("2025-08-01"),
    top_k=3,
)

print("Retrieved:", len(chunks))
for c in chunks:
    print(c.doc_id, c.source, c.filed, c.text[:120])