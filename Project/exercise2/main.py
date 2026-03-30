#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 2 – NVDA Valuation Agent
Runs two versions of the agent (with / without sentiment) over two TTM windows,
then compares Buy/Sell/Hold signals, confidence, and Sharpe Ratios.
"""
from __future__ import annotations

import os
import math
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from llm_backend import OpenAIBackend
from sec_fundamentals import SecConfig, build_quarter_table
from market_data import get_price_series
from valuation_agent import ValuationAgent, ValuationInputs, ValuationAgentConfig
from backtester import EventBacktester, BacktestConfig, compute_sharpe
from filing_rag import FilingRAG


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKER      = "NVDA"
PRICE_START = "2023-01-01"
PRICE_END   = "2025-01-01"

EVENT_WINDOWS = {
    "window_feb": pd.Timestamp("2024-02-07"),
    "window_aug": pd.Timestamp("2024-08-07"),
}

SENTIMENT = {
    "window_feb": {"lm": 0.0748, "finbert": 0.3265, "gpt": 6.81},
    "window_aug": {"lm": 0.0316, "finbert": 0.3650, "gpt": 6.61},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def align_to_trading_day(prices: pd.Series, dt: pd.Timestamp) -> pd.Timestamp:
    idx = prices.index
    if dt in idx:
        return dt
    later = idx[idx >= dt]
    return pd.Timestamp(later[0]) if len(later) > 0 else pd.Timestamp(idx[-1])


def get_recent_prices(prices: pd.Series, asof: pd.Timestamp, n: int = 5):
    past = prices[prices.index <= asof]
    return [float(v) for v in past.tail(n).values.flatten()]


def get_pit_shares(quarter_table: pd.DataFrame, asof: pd.Timestamp) -> Optional[float]:
    """
    Return the most recent point-in-time shares_outstanding from the quarter
    table where the quarter end date <= asof.  Uses the SEC-reported value
    directly (already on a post-split basis for all dates because SEC filers
    retroactively restate share counts after splits, and yfinance prices are
    also split-adjusted, so the two are consistent).
    """
    if "shares_outstanding" not in quarter_table.columns:
        return None
    qt = quarter_table.copy()
    qt["end"] = pd.to_datetime(qt["end"], errors="coerce")
    avail = qt[qt["end"] <= asof]["shares_outstanding"].dropna()
    if avail.empty:
        return None
    return float(avail.iloc[-1])


def build_vin_map(ticker, prices, quarter_table, window_name, event_dt, use_sentiment):
    trade_dt  = align_to_trading_day(prices, event_dt)
    price_val = prices.loc[trade_dt]
    price     = float(price_val.iloc[0]) if hasattr(price_val, "iloc") else float(price_val)
    sentiment = SENTIMENT[window_name] if use_sentiment else {}
    pit_shares = get_pit_shares(quarter_table, trade_dt)

    from sec_fundamentals import ttm_from_quarters as _ttm
    q = quarter_table.copy()
    q["end"]   = pd.to_datetime(q["end"])
    q["filed"] = pd.to_datetime(q["filed"], errors="coerce")
    eligible   = q[q["filed"].notna() & (q["filed"] <= trade_dt)]
    last_end   = eligible["end"].max()
    q2         = q[q["end"] <= last_end].tail(4)
    print(f"\n  [TTM quarters for {window_name}]")
    print(q2[["end","filed","revenue","net_income","fcf"]].to_string())

    vin = ValuationInputs(
        asof=trade_dt,
        ticker=ticker,
        price=price,
        market_cap=None,
        shares_outstanding=pit_shares,
        quarter_table=quarter_table,
        sentiment_lm=sentiment.get("lm") if use_sentiment else None,
        sentiment_finbert=sentiment.get("finbert") if use_sentiment else None,
        sentiment_gpt=sentiment.get("gpt") if use_sentiment else None,
        recent_prices=get_recent_prices(prices, trade_dt, n=5),
    )
    return {trade_dt: vin}


def run_agent(label, ticker, prices, quarter_table, agent, use_sentiment):
    print(f"\n{'='*60}\n  Running: {label}\n{'='*60}")

    combined_vin_map = {}
    trade_dts = {}
    for wname, event_dt in EVENT_WINDOWS.items():
        vin_map  = build_vin_map(ticker, prices, quarter_table, wname, event_dt, use_sentiment)
        trade_dt = list(vin_map.keys())[0]
        combined_vin_map.update(vin_map)
        trade_dts[wname] = trade_dt

    bt = EventBacktester(
        prices=prices,
        cfg=BacktestConfig(
            initial_cash=100_000.0,
            transaction_cost_bps=10.0,
            trade_size_units=5.0,
            allow_short=True,
        ),
    )
    results = bt.run(ticker=ticker, agent=agent,
                     valuation_inputs_by_event_date=combined_vin_map)
    sharpe  = compute_sharpe(results)
    cum_ret = results["portfolio_value"].iloc[-1] / results["portfolio_value"].iloc[0] - 1

    decisions = {}
    for wname, trade_dt in trade_dts.items():
        row = results.loc[trade_dt] if trade_dt in results.index else results.iloc[0]
        decisions[wname] = {
            "trade_dt":   trade_dt,
            "price":      row["price"],
            "action":     row["action"],
            "confidence": row["decision_confidence"],
            "score":      row["decision_score"],
            "thesis":     row["decision_thesis"],
            "sharpe":     sharpe,
            "cum_return": cum_ret,
        }
        print(f"\n  [{wname}] date={trade_dt.date()}  price={row['price']:.2f}")
        print(f"  Action={row['action'].upper()}  "
              f"Confidence={row['decision_confidence']}  Score={row['decision_score']}")
        print(f"  Thesis: {row['decision_thesis']}")

    sharpe_str = f"{sharpe:.4f}" if not math.isnan(sharpe) else "N/A (no trades)"
    print(f"\n  Overall Sharpe={sharpe_str}  CumReturn={cum_ret:.2%}")
    return decisions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY first.")

    sec_cfg = SecConfig(user_agent="ValuationAgent xyz@gmail.com")

    print(f"Fetching price data for {TICKER}...")
    prices = get_price_series(TICKER, start=PRICE_START, end=PRICE_END)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().astype(float)
    prices = prices[~prices.index.duplicated(keep="first")]

    print("Fetching SEC fundamentals...")
    quarter_table = build_quarter_table(TICKER, sec_cfg)
    print(f"Quarter table ready: {len(quarter_table)} rows")
    print(quarter_table[["end", "revenue", "net_income", "fcf"]].tail(6).to_string())

    llm = OpenAIBackend(
        chat_model="gpt-4o",
        embed_model="text-embedding-3-small",
        temperature=0.2,
        max_output_tokens=800,
    )

    agent_cfg = ValuationAgentConfig(
        use_llm=True,
        cheap_threshold_ps=10.0,    expensive_threshold_ps=30.0,
        cheap_threshold_pe=30.0,    expensive_threshold_pe=70.0,
        cheap_threshold_pfcf=25.0,  expensive_threshold_pfcf=60.0,
    )
    agent = ValuationAgent(llm=llm, config=agent_cfg, filing_rag=FilingRAG())

    dec_no  = run_agent("Agent WITHOUT Sentiment", TICKER, prices,
                        quarter_table, agent, use_sentiment=False)
    dec_yes = run_agent("Agent WITH Sentiment",    TICKER, prices,
                        quarter_table, agent, use_sentiment=True)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  FINAL COMPARISON: With vs Without Sentiment")
    print("=" * 72)
    fmt = "{:<15} {:<25} {:<8} {:<8} {:<20} {}"
    print(fmt.format("Window", "Version", "Action", "Conf", "Sharpe", "CumReturn"))
    print("-" * 72)
    for wn in EVENT_WINDOWS:
        d0, d1 = dec_no[wn], dec_yes[wn]
        s0 = f"{d0['sharpe']:.4f}" if not math.isnan(d0["sharpe"]) else "N/A (no trades)"
        s1 = f"{d1['sharpe']:.4f}" if not math.isnan(d1["sharpe"]) else "N/A (no trades)"
        print(fmt.format(wn, "WITHOUT sentiment",
                         d0["action"].upper(), str(d0["confidence"]),
                         s0, f"{d0['cum_return']:.2%}"))
        print(fmt.format(wn, "WITH sentiment",
                         d1["action"].upper(), str(d1["confidence"]),
                         s1, f"{d1['cum_return']:.2%}"))
        print()

    print("\nThesis comparison:")
    for wn in EVENT_WINDOWS:
        print(f"\n[{wn}]")
        print(f"  NO  sentiment: {dec_no[wn]['thesis']}")
        print(f"  YES sentiment: {dec_yes[wn]['thesis']}")


if __name__ == "__main__":
    main()
