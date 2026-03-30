#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
valuation_agent.py  —  Modified for Exercise 2
Changes vs original:
  - ValuationInputs now accepts sentiment_scores (dict) and recent_prices (list)
  - ValuationAgentConfig has use_sentiment flag
  - decide() builds two prompt variants: with / without sentiment
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd

from llm_backend import LLMBackend
from sec_fundamentals import ttm_from_quarters
from filing_rag import FilingRAG


@dataclass
class ValuationInputs:
    asof: pd.Timestamp
    ticker: str
    price: float
    market_cap: Optional[float]
    shares_outstanding: Optional[float]
    quarter_table: pd.DataFrame
    # NEW: Exercise 2 additions
    sentiment_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    # e.g. {"lm_score": 0.075, "finbert_score": 0.327, "gpt_score": 6.81}
    recent_prices: List[float] = field(default_factory=list)
    # last 5 trading-day closing prices, oldest first


@dataclass
class ValuationDecision:
    ticker: str
    asof: pd.Timestamp
    action: str
    confidence: float
    score: float
    thesis: str
    key_points: List[str]
    risks: List[str]
    metrics: Dict[str, Optional[float]]


@dataclass
class ValuationAgentConfig:
    output_actions: Tuple[str, ...] = ("buy", "sell", "hold")
    horizon: str = "medium"
    cheap_threshold_ps: float = 3.0
    expensive_threshold_ps: float = 8.0
    cheap_threshold_pe: float = 15.0
    expensive_threshold_pe: float = 30.0
    cheap_threshold_pfcf: float = 12.0
    expensive_threshold_pfcf: float = 25.0
    use_llm: bool = True
    rag_top_k: int = 4
    # NEW
    use_sentiment: bool = False


class ValuationAgent:
    def __init__(self, llm: LLMBackend, config=None, filing_rag=None):
        self.llm = llm
        self.cfg = config or ValuationAgentConfig()
        self.filing_rag = filing_rag

    @staticmethod
    def _safe_div(a, b):
        if a is None or b is None or abs(b) < 1e-12:
            return None
        return float(a / b)

    def compute_metrics(self, vin: ValuationInputs):
        q = vin.quarter_table.dropna(subset=["end"]).sort_values("end")
        if q.empty:
            return {"ttm_revenue": None, "ttm_op_income": None,
                    "ttm_net_income": None, "ttm_fcf": None,
                    "shares_outstanding": None, "market_cap": None,
                    "ps": None, "pe": None, "p_fcf": None}

        eligible = q[q["filed"].notna() & (q["filed"] <= vin.asof)]
        if eligible.empty:
            eligible = q
        last_end = eligible["end"].max()
        ttm = ttm_from_quarters(q, pd.Timestamp(last_end))

        shares = ttm.get("shares_outstanding") or vin.shares_outstanding
        market_cap = None
        if shares is not None:
            market_cap = float(vin.price) * float(shares)
        elif vin.market_cap is not None:
            market_cap = float(vin.market_cap)

        ps    = self._safe_div(market_cap, ttm["ttm_revenue"])
        pe    = self._safe_div(market_cap, ttm["ttm_net_income"])
        p_fcf = self._safe_div(market_cap, ttm["ttm_fcf"])
        return {**ttm, "market_cap": market_cap, "ps": ps, "pe": pe, "p_fcf": p_fcf}

    def _rule_prior(self, metrics):
        score, n = 0.0, 0
        ps = metrics.get("ps")
        if isinstance(ps, (int, float)):
            n += 1
            if ps <= self.cfg.cheap_threshold_ps: score += 0.35
            elif ps >= self.cfg.expensive_threshold_ps: score -= 0.35
        pe = metrics.get("pe")
        if isinstance(pe, (int, float)):
            n += 1
            if pe <= self.cfg.cheap_threshold_pe: score += 0.35
            elif pe >= self.cfg.expensive_threshold_pe: score -= 0.35
        p_fcf = metrics.get("p_fcf")
        if isinstance(p_fcf, (int, float)):
            n += 1
            if p_fcf <= self.cfg.cheap_threshold_pfcf: score += 0.30
            elif p_fcf >= self.cfg.expensive_threshold_pfcf: score -= 0.30
        return 0.0 if n == 0 else max(-1.0, min(1.0, float(score)))

    @staticmethod
    def _parse_key_lines(raw):
        keys = ["ACTION", "CONFIDENCE", "SCORE", "THESIS", "POINTS", "RISKS"]
        parsed = {k: "" for k in keys}
        for line in (raw or "").splitlines():
            if ":" not in line: continue
            k, v = line.split(":", 1)
            k = k.strip().upper()
            if k in parsed: parsed[k] = v.strip()
        return parsed

    @staticmethod
    def _to_float(x, default):
        try: return float(x)
        except: return default

    def _retrieve_filing_context(self, vin):
        if self.filing_rag is None:
            return "(no filing RAG attached)"
        chunks = self.filing_rag.retrieve(
            self.llm, ticker=vin.ticker,
            query=f"{vin.ticker} valuation, margins, growth, risks, cash flow, capital allocation, guidance",
            asof=vin.asof, top_k=self.cfg.rag_top_k)
        if not chunks: return "(no filing text retrieved)"
        return "\n\n".join(
            f"[{c.source} filed={c.filed.date()} doc={c.doc_id}] {c.text}" for c in chunks)

    def _sentiment_block(self, vin):
        if not self.cfg.use_sentiment: return ""
        s = vin.sentiment_scores
        if not s: return ""
        return (
            "\nEarnings-call sentiment (from Exercise 1 NLP pipeline):\n"
            f"  - LM lexicon score : {s.get('lm_score', 'N/A')}  (positive net word ratio; range ~-1 to +1)\n"
            f"  - FinBERT score    : {s.get('finbert_score', 'N/A')}  (fine-tuned financial BERT; range 0 to +1)\n"
            f"  - GPT intensity    : {s.get('gpt_score', 'N/A')}  (LLM-rated forward-looking intensity; range 0 to 10)\n"
            "Higher values = more positive / bullish management tone."
        )

    @staticmethod
    def _prices_block(vin):
        if not vin.recent_prices: return ""
        prices_str = ", ".join(f"${p:.2f}" for p in vin.recent_prices)
        change = ""
        if len(vin.recent_prices) >= 2:
            pct = (vin.recent_prices[-1] / vin.recent_prices[0] - 1) * 100
            change = f"  (5-day change: {pct:+.1f}%)"
        return f"\nLast 5 trading-day closing prices (oldest to newest): {prices_str}{change}"

    def decide(self, vin: ValuationInputs) -> ValuationDecision:
        metrics        = self.compute_metrics(vin)
        prior          = self._rule_prior(metrics)
        filing_context = self._retrieve_filing_context(vin)
        sentiment_text = self._sentiment_block(vin)
        prices_text    = self._prices_block(vin)

        if getattr(self.cfg, "use_llm", True) is False:
            action = "buy" if prior > 0.25 else ("sell" if prior < -0.25 else "hold")
            return ValuationDecision(
                ticker=vin.ticker, asof=vin.asof, action=action,
                confidence=0.55, score=prior,
                thesis="LLM disabled: using rule-based valuation prior only.",
                key_points=[f"P/S={metrics.get('ps')}", f"P/E={metrics.get('pe')}", f"P/FCF={metrics.get('p_fcf')}"],
                risks=["LLM disabled; qualitative context missing."],
                metrics=metrics)

        sentiment_label = "WITH sentiment indicator" if self.cfg.use_sentiment else "WITHOUT sentiment indicator"
        sentiment_instruction = (
            "\nAlso explicitly incorporate the earnings-call sentiment scores above into your thesis."
            if self.cfg.use_sentiment else "")

        prompt = f"""You are a valuation-focused equity analyst. This analysis runs {sentiment_label}.

Ticker     : {vin.ticker}
As-of date : {vin.asof.date()}
Price      : ${vin.price:.2f}
Horizon    : {self.cfg.horizon}
{prices_text}

--- Point-in-time TTM valuation metrics ---
Market Cap          : {metrics.get('market_cap')}
TTM Revenue         : {metrics.get('ttm_revenue')}
TTM Operating Income: {metrics.get('ttm_op_income')}
TTM Net Income      : {metrics.get('ttm_net_income')}
TTM Free Cash Flow  : {metrics.get('ttm_fcf')}
Shares Outstanding  : {metrics.get('shares_outstanding')}
P/S                 : {metrics.get('ps')}
P/E                 : {metrics.get('pe')}
P/FCF               : {metrics.get('p_fcf')}

Rule-based valuation prior (range -1 to +1): {prior}
{sentiment_text}

--- Retrieved filing context ---
{filing_context}

Task:
Assess whether {vin.ticker} is undervalued, fairly valued, or overvalued.
Base your reasoning on valuation multiples, cash generation, profitability,
and price momentum.{sentiment_instruction}
Be conservative and numerate.

Output exactly (no extra lines):
ACTION: <buy|sell|hold>
CONFIDENCE: <0..1 float>
SCORE: <-1..+1 float>
THESIS: <1-3 sentences>
POINTS: <semicolon-separated key positives/negatives>
RISKS: <semicolon-separated key risks>""".strip()

        resp = self.llm.chat([
            {"role": "system", "content": "Be precise, numerate, and conservative."},
            {"role": "user",   "content": prompt}])

        raw = (resp.get("content") or "").strip()
        p   = self._parse_key_lines(raw)

        action = p["ACTION"].lower()
        if action not in self.cfg.output_actions:
            action = "buy" if prior > 0.25 else ("sell" if prior < -0.25 else "hold")

        return ValuationDecision(
            ticker=vin.ticker, asof=vin.asof,
            action=action,
            confidence=max(0.0, min(1.0, self._to_float(p["CONFIDENCE"], 0.55))),
            score=max(-1.0, min(1.0, self._to_float(p["SCORE"], prior))),
            thesis=p["THESIS"] or "Valuation signal is weak or incomplete; staying neutral.",
            key_points=[x.strip() for x in (p["POINTS"] or "").split(";") if x.strip()],
            risks=[x.strip() for x in (p["RISKS"] or "").split(";") if x.strip()],
            metrics=metrics)
